use std::error::Error;
use std::fs;
use std::path::PathBuf;

use burn::module::Module;
use burn::store::{BurnpackStore, ModuleSnapshot};
use wired_policy::ActorCritic;
use wired_sim::{FlyModel, TaskConfig};

use crate::ppo;

const DEFAULT_MODEL: &str = "data/flybody/fruitfly.xml";

pub fn run(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn Error>> {
    let mut model_path = PathBuf::from(DEFAULT_MODEL);
    let mut policy_path = ppo::default_policy();
    let mut output_dir = PathBuf::from("results/frames");
    let mut seconds = 5.0_f64;
    let mut fps = 30.0_f64;
    let mut width = 640_usize;
    let mut height = 480_usize;
    let mut cpu = false;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => model_path = PathBuf::from(value(&mut args, "--model")?),
            "--policy" => policy_path = PathBuf::from(value(&mut args, "--policy")?),
            "--output" => output_dir = PathBuf::from(value(&mut args, "--output")?),
            "--seconds" => seconds = value(&mut args, "--seconds")?.parse()?,
            "--fps" => fps = value(&mut args, "--fps")?.parse()?,
            "--width" => width = value(&mut args, "--width")?.parse()?,
            "--height" => height = value(&mut args, "--height")?.parse()?,
            "--cpu" => cpu = true,
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }
    if seconds <= 0.0 || fps <= 0.0 || width == 0 || height == 0 {
        return Err("seconds, fps, width and height must be positive".into());
    }

    let device = ppo::select_device(cpu)?;
    let model = FlyModel::load_tethered(&model_path)?;
    let config = TaskConfig::default();
    let mut env = model.tethered_env(config)?;
    env.reset();
    let observation_dim = env.observation_dim();
    let action_dim = env.action_dim();

    let mut policy = ActorCritic::new(observation_dim, 256, action_dim, &device);
    let mut store = BurnpackStore::from_file(&policy_path);
    let loaded = policy
        .load_from(&mut store)
        .map_err(|error| format!("could not load {}: {error}", policy_path.display()))?;
    if !loaded.is_success() {
        return Err(format!("incomplete policy checkpoint: {loaded:?}").into());
    }
    let valid = policy.valid();

    let mut renderer = env.frame_renderer(width, height)?;
    fs::create_dir_all(&output_dir)?;

    let control_period = model.timestep() * config.control_decimation as f64;
    let steps = (seconds / control_period).round().max(1.0) as usize;
    let stride = (1.0 / (fps * control_period)).round().max(1.0) as usize;
    let mut frame = 0_usize;
    let mut total_reward = 0.0_f64;
    let mut ball_y = 0.0_f64;

    for step in 0..steps {
        let observation = env.observation();
        let tensor = ppo::rows_to_tensor(&[observation], observation_dim, &device);
        let (mean, _) = valid.forward(tensor);
        let action = ppo::tensor_to_vec(mean)
            .into_iter()
            .map(|value| value.tanh())
            .collect::<Vec<f32>>();

        let result = env.step(&action)?;
        total_reward += result.reward;
        ball_y += env.ball_velocity()[1];

        if step % stride == 0 {
            let path = output_dir.join(format!("frame_{frame:04}.png"));
            env.render_frame(&mut renderer, &path)?;
            frame += 1;
        }
    }

    println!(
        "rendered {frame} frames to {} ({:.1}s at {fps} fps)",
        output_dir.display(),
        frame as f64 / fps
    );
    println!(
        "mean_reward={:.4} mean_ball_y_velocity={:.4}",
        total_reward / steps as f64,
        ball_y / steps as f64
    );
    Ok(())
}

fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
