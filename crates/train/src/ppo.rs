use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::TensorData;
use rand::Rng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use wired_policy::ActorCritic;
use wired_sim::{FlyModel, TaskConfig, TetheredEnv};

const LN_2PI: f32 = 1.837_877_1;
const LN_2PI_E: f32 = 2.837_877;

#[derive(Debug, Clone, Serialize)]
pub struct PpoConfig {
    pub updates: usize,
    pub envs: usize,
    pub rollout_steps: usize,
    pub epochs: usize,
    pub minibatch: usize,
    pub learning_rate: f64,
    pub gamma: f64,
    pub lambda: f64,
    pub clip: f64,
    pub value_coef: f64,
    pub entropy_coef: f64,
    pub seed: u64,
    pub control_decimation: usize,
    pub episode_steps: usize,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            updates: 100,
            envs: 8,
            rollout_steps: 64,
            epochs: 4,
            minibatch: 256,
            learning_rate: 3e-4,
            gamma: 0.99,
            lambda: 0.95,
            clip: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.001,
            seed: 7,
            control_decimation: 20,
            episode_steps: 500,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateLog {
    pub update: usize,
    pub mean_reward: f32,
    pub mean_ball_y_velocity: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalResult {
    pub mean_reward: f32,
    pub mean_ball_y_velocity: f32,
    pub mean_ball_y_error: f32,
    pub steps: usize,
}

#[derive(Debug, Serialize)]
pub struct PpoReport {
    pub config: PpoConfig,
    pub initial_eval: EvalResult,
    pub final_eval: EvalResult,
    pub updates: Vec<UpdateLog>,
}

type Rollout = (Vec<Vec<Transition>>, f32, f32);

struct Transition {
    observation: Vec<f32>,
    action: Vec<f32>,
    log_prob: f32,
    value: f32,
    reward: f32,
    done: bool,
    advantage: f32,
    return_: f32,
}

pub fn run(
    model_path: &Path,
    output_path: &Path,
    config: PpoConfig,
    cpu: bool,
) -> Result<(), Box<dyn Error>> {
    let device = if cpu { Device::flex() } else { Device::cuda(0) }.autodiff();

    let model = FlyModel::load_tethered(model_path)?;
    let task_config = TaskConfig {
        control_decimation: config.control_decimation,
        episode_control_steps: config.episode_steps,
        target_height: 0.0,
    };
    let observation_dim = model.tethered_env(task_config)?.observation_dim();
    let action_dim = model.action_dim();

    let mut envs = (0..config.envs)
        .map(|_| model.tethered_env(task_config))
        .collect::<Result<Vec<_>, _>>()?;
    for env in &mut envs {
        env.reset();
    }

    let mut policy = ActorCritic::new(observation_dim, 256, action_dim, &device);
    let mut optimizer = AdamConfig::new().init();
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

    println!(
        "ppo: device={device:?} envs={} rollout={} updates={} observation_dim={observation_dim} action_dim={action_dim} params={}",
        config.envs,
        config.rollout_steps,
        config.updates,
        policy.parameter_count()
    );

    let initial_eval = evaluate(&policy, &model, task_config)?;
    let mut logs = Vec::with_capacity(config.updates);

    for update in 0..config.updates {
        let (transitions, mean_reward, mean_ball_y) =
            collect_rollout(&policy, &mut envs, &mut rng, &config, &device)?;

        let (policy_loss, value_loss, entropy) = train_epochs(
            &mut policy,
            &mut optimizer,
            &transitions,
            &config,
            &mut rng,
            &device,
        );

        let log = UpdateLog {
            update: update + 1,
            mean_reward,
            mean_ball_y_velocity: mean_ball_y,
            policy_loss,
            value_loss,
            entropy,
        };
        println!(
            "update {:>4}/{} reward={:.4} ball_y={:.4} pi_loss={:.4} v_loss={:.4} entropy={:.4}",
            log.update,
            config.updates,
            log.mean_reward,
            log.mean_ball_y_velocity,
            log.policy_loss,
            log.value_loss,
            log.entropy
        );
        logs.push(log);
    }

    let final_eval = evaluate(&policy, &model, task_config)?;
    println!(
        "eval: initial_reward={:.4} final_reward={:.4} final_ball_y={:.4} final_error={:.4}",
        initial_eval.mean_reward,
        final_eval.mean_reward,
        final_eval.mean_ball_y_velocity,
        final_eval.mean_ball_y_error
    );

    let report = PpoReport {
        config,
        initial_eval,
        final_eval,
        updates: logs,
    };
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output_path, serde_json::to_vec_pretty(&report)?)?;
    println!("wrote {}", output_path.display());

    Ok(())
}

fn collect_rollout(
    policy: &ActorCritic,
    envs: &mut [TetheredEnv],
    rng: &mut ChaCha8Rng,
    config: &PpoConfig,
    device: &Device,
) -> Result<Rollout, Box<dyn Error>> {
    let n = envs.len();
    let observation_dim = envs[0].observation_dim();
    let action_dim = envs[0].action_dim();

    let valid = policy.valid();
    let log_std = tensor_to_vec(valid.log_std());
    let mut trajectories: Vec<Vec<Transition>> = (0..n)
        .map(|_| Vec::with_capacity(config.rollout_steps))
        .collect();
    let mut reward_sum = 0.0_f32;
    let mut ball_y_sum = 0.0_f32;

    for _ in 0..config.rollout_steps {
        let observations: Vec<Vec<f32>> = envs.iter().map(TetheredEnv::observation).collect();
        let observation_tensor = rows_to_tensor(&observations, observation_dim, device);
        let (mean, value) = valid.forward(observation_tensor);
        let mean_values = tensor_to_vec(mean);
        let values = tensor_to_vec(value);

        for (index, env) in envs.iter_mut().enumerate() {
            let mut action = Vec::with_capacity(action_dim);
            let mut log_prob = 0.0_f32;
            for (action_index, log_std_value) in log_std.iter().enumerate().take(action_dim) {
                let offset = index * action_dim + action_index;
                let mean_value = mean_values[offset];
                let std = log_std_value.exp();
                let z = gaussian(rng);
                let sampled = (mean_value + std * z).clamp(-1.0, 1.0);
                let normalized = (sampled - mean_value) / std;
                log_prob += -0.5 * normalized * normalized - log_std_value - 0.5 * LN_2PI;
                action.push(sampled);
            }

            let step = env.step(&action)?;
            reward_sum += step.reward as f32;
            ball_y_sum += env.ball_velocity()[1] as f32;
            trajectories[index].push(Transition {
                observation: observations[index].clone(),
                action,
                log_prob,
                value: values[index],
                reward: step.reward as f32,
                done: step.done,
                advantage: 0.0,
                return_: 0.0,
            });
            if step.done {
                env.reset();
            }
        }
    }

    let final_observations: Vec<Vec<f32>> = envs.iter().map(TetheredEnv::observation).collect();
    let final_tensor = rows_to_tensor(&final_observations, observation_dim, device);
    let (_, final_value) = valid.forward(final_tensor);
    let final_values = tensor_to_vec(final_value);

    for (env_index, trajectory) in trajectories.iter_mut().enumerate() {
        let mut gae = 0.0_f32;
        for step in (0..trajectory.len()).rev() {
            let next_value = if step + 1 < trajectory.len() {
                trajectory[step + 1].value
            } else {
                final_values[env_index]
            };
            let mask = if trajectory[step].done { 0.0 } else { 1.0 };
            let delta = trajectory[step].reward + config.gamma as f32 * next_value * mask
                - trajectory[step].value;
            gae = delta + config.gamma as f32 * config.lambda as f32 * mask * gae;
            trajectory[step].advantage = gae;
            trajectory[step].return_ = gae + trajectory[step].value;
        }
    }

    let count = (n * config.rollout_steps) as f32;
    Ok((trajectories, reward_sum / count, ball_y_sum / count))
}

fn train_epochs(
    policy: &mut ActorCritic,
    optimizer: &mut burn::optim::ModuleOptimizer,
    trajectories: &[Vec<Transition>],
    config: &PpoConfig,
    rng: &mut ChaCha8Rng,
    device: &Device,
) -> (f32, f32, f32) {
    let flat: Vec<&Transition> = trajectories.iter().flatten().collect();
    let advantages: Vec<f32> = flat.iter().map(|transition| transition.advantage).collect();
    let (advantage_mean, advantage_std) = mean_std(&advantages);
    let mut order: Vec<usize> = (0..flat.len()).collect();
    let observation_dim = flat[0].observation.len();
    let action_dim = flat[0].action.len();

    let mut last_policy_loss = 0.0;
    let mut last_value_loss = 0.0;
    let mut last_entropy = 0.0;

    for _ in 0..config.epochs {
        order.shuffle(rng);
        for chunk in order.chunks(config.minibatch) {
            let observations: Vec<Vec<f32>> = chunk
                .iter()
                .map(|index| flat[*index].observation.clone())
                .collect();
            let actions: Vec<Vec<f32>> = chunk
                .iter()
                .map(|index| flat[*index].action.clone())
                .collect();
            let old_log_probs: Vec<Vec<f32>> = chunk
                .iter()
                .map(|index| vec![flat[*index].log_prob])
                .collect();
            let returns: Vec<Vec<f32>> = chunk
                .iter()
                .map(|index| vec![flat[*index].return_])
                .collect();
            let normalized_advantages: Vec<Vec<f32>> = chunk
                .iter()
                .map(|index| {
                    vec![(flat[*index].advantage - advantage_mean) / (advantage_std + 1e-8)]
                })
                .collect();

            let observation_tensor = rows_to_tensor(&observations, observation_dim, device);
            let action_tensor = rows_to_tensor(&actions, action_dim, device);
            let old_log_prob_tensor = rows_to_tensor(&old_log_probs, 1, device);
            let return_tensor = rows_to_tensor(&returns, 1, device);
            let advantage_tensor = rows_to_tensor(&normalized_advantages, 1, device);

            let (mean, value) = policy.forward(observation_tensor);
            let log_std = policy.log_std();
            let normalized = action_tensor.sub(mean).div(log_std.clone().exp());
            let new_log_prob = normalized
                .powi_scalar(2)
                .mul_scalar(-0.5)
                .sub(log_std.clone())
                .sub_scalar(0.5 * LN_2PI);
            let new_log_prob = new_log_prob.sum_dim(1);

            let ratio = new_log_prob.sub(old_log_prob_tensor).exp();
            let clipped_ratio = ratio
                .clone()
                .clamp(1.0 - config.clip as f32, 1.0 + config.clip as f32);
            let policy_loss = ratio
                .mul(advantage_tensor.clone())
                .min_pair(clipped_ratio.mul(advantage_tensor))
                .mean()
                .neg();
            let value_loss = value.sub(return_tensor).powi_scalar(2).mean();
            let entropy = log_std.clone().add_scalar(0.5 * LN_2PI_E).sum_dim(1).mean();
            let loss = policy_loss
                .clone()
                .add(value_loss.clone().mul_scalar(config.value_coef as f32))
                .sub(entropy.clone().mul_scalar(config.entropy_coef as f32));

            let gradients = GradientsParams::from_grads(loss.backward(), policy);
            *policy = optimizer.step(config.learning_rate, policy.clone(), gradients);

            last_policy_loss = tensor_scalar(policy_loss);
            last_value_loss = tensor_scalar(value_loss);
            last_entropy = tensor_scalar(entropy);
        }
    }

    (last_policy_loss, last_value_loss, last_entropy)
}

fn evaluate(
    policy: &ActorCritic,
    model: &FlyModel,
    task_config: TaskConfig,
) -> Result<EvalResult, Box<dyn Error>> {
    let mut env = model.tethered_env(task_config)?;
    env.reset();
    let valid = policy.valid();
    let device = valid.log_std().device();
    let observation_dim = env.observation_dim();
    let mut total_reward = 0.0_f32;
    let mut ball_y_sum = 0.0_f32;
    let mut error_sum = 0.0_f32;
    let mut steps = 0;

    loop {
        let observation = rows_to_tensor(&[env.observation()], observation_dim, &device);
        let (mean, _) = valid.forward(observation);
        let action = tensor_to_vec(mean)
            .into_iter()
            .map(|value| value.clamp(-1.0, 1.0))
            .collect::<Vec<f32>>();
        let result = env.step(&action)?;
        total_reward += result.reward as f32;
        ball_y_sum += env.ball_velocity()[1] as f32;
        error_sum += (env.ball_velocity()[1] + 5.0).abs() as f32;
        steps += 1;
        if result.done {
            break;
        }
    }

    Ok(EvalResult {
        mean_reward: total_reward / steps as f32,
        mean_ball_y_velocity: ball_y_sum / steps as f32,
        mean_ball_y_error: error_sum / steps as f32,
        steps,
    })
}

fn rows_to_tensor(rows: &[Vec<f32>], width: usize, device: &Device) -> Tensor<2> {
    let mut flat = Vec::with_capacity(rows.len() * width);
    for row in rows {
        debug_assert_eq!(row.len(), width);
        flat.extend_from_slice(row);
    }
    Tensor::<2>::from_data(TensorData::new(flat, vec![rows.len(), width]), device)
}

fn tensor_to_vec<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .to_data()
        .try_to_vec::<f32>()
        .expect("tensor must be readable as f32")
}

fn tensor_scalar(tensor: Tensor<1>) -> f32 {
    tensor_to_vec(tensor).first().copied().unwrap_or_default()
}

fn gaussian(rng: &mut ChaCha8Rng) -> f32 {
    let u1: f32 = rng.random_range(1e-6..1.0);
    let u2: f32 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 1.0);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32;
    (mean, variance.sqrt().max(1e-6))
}

pub fn default_output() -> PathBuf {
    PathBuf::from("results/ppo.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_std_is_stable_for_constant_values() {
        let (mean, std) = mean_std(&[2.0, 2.0, 2.0]);
        assert_eq!(mean, 2.0);
        assert!(std > 0.0);
    }
}
