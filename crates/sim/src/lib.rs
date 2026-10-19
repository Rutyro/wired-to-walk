//! MuJoCo fly environment used by the training loop and the throughput spike.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use mujoco_rs::prelude::{MjData, MjModel, MjtObj, MjtTrn};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("could not load MuJoCo model {path}: {detail}")]
    Model { path: String, detail: String },
    #[error("MuJoCo data error: {0}")]
    Data(String),
    #[error("expected {expected} controls, got {actual}")]
    ActionSize { expected: usize, actual: usize },
}

#[derive(Debug, Clone, Copy)]
struct Control {
    index: usize,
    min: f64,
    max: f64,
    neutral: f64,
    scale: f64,
}

pub struct FlyModel {
    model: Arc<MjModel>,
    timestep: f64,
    controls: Vec<Control>,
    wing_qpos: Vec<(usize, f64)>,
    joint_positions: usize,
    joint_velocities: usize,
    observation_dim: usize,
    has_free_root: bool,
}

impl FlyModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SimError> {
        let path = path.as_ref();
        let model = MjModel::from_xml(path).map_err(|error| SimError::Model {
            path: path.display().to_string(),
            detail: error.to_string(),
        })?;
        let timestep = model.opt().timestep;
        let nq = model.nq() as usize;
        let nv = model.nv() as usize;
        let controls = build_controls(&model);
        let wing_qpos = wing_spring_positions(&model);

        if timestep <= 0.0 {
            return Err(SimError::Model {
                path: path.display().to_string(),
                detail: "timestep must be positive".to_owned(),
            });
        }
        if controls.is_empty() {
            return Err(SimError::Model {
                path: path.display().to_string(),
                detail: "model has no walking actuators (head, abdomen or legs)".to_owned(),
            });
        }
        let has_free_root = model.name_to_id(MjtObj::mjOBJ_JOINT, "free").is_some();
        let joint_positions = nq.saturating_sub(7);
        let joint_velocities = nv.saturating_sub(6);
        let observation_dim = if has_free_root {
            3 + 3 + 1 + 2 + 1 + joint_positions + joint_velocities + controls.len()
        } else {
            0
        };

        Ok(Self {
            model: Arc::new(model),
            timestep,
            controls,
            wing_qpos,
            joint_positions,
            joint_velocities,
            observation_dim,
            has_free_root,
        })
    }

    /// Build a tethered walking model: the thorax is fixed to the world and a
    /// floating ball is placed under the claws, matching flybody's
    /// `walk_on_ball` task.
    pub fn load_tethered(path: impl AsRef<Path>) -> Result<Self, SimError> {
        let path = path.as_ref();
        let output = path.with_file_name("wired_tethered.xml");
        prepare_tethered_model(path, &output)?;
        Self::load(&output)
    }

    pub fn world(&self) -> Result<FlyWorld, SimError> {
        Ok(FlyWorld {
            data: MjData::try_new(Arc::clone(&self.model))
                .map_err(|error| SimError::Data(error.to_string()))?,
            controls: self.controls.clone(),
            wing_qpos: self.wing_qpos.clone(),
        })
    }

    pub fn timestep(&self) -> f64 {
        self.timestep
    }

    pub fn action_dim(&self) -> usize {
        self.controls.len()
    }

    pub fn observation_dim(&self) -> usize {
        self.observation_dim
    }

    pub fn has_free_root(&self) -> bool {
        self.has_free_root
    }

    pub fn qpos_dim(&self) -> usize {
        self.model.nq() as usize
    }

    pub fn qvel_dim(&self) -> usize {
        self.model.nv() as usize
    }

    pub fn body_count(&self) -> usize {
        self.model.nbody() as usize
    }

    pub fn dof_count(&self) -> usize {
        self.model.nv() as usize
    }

    pub fn joint_positions(&self) -> usize {
        self.joint_positions
    }

    pub fn joint_velocities(&self) -> usize {
        self.joint_velocities
    }

    pub fn env(&self, config: TaskConfig) -> Result<FlyEnv, SimError> {
        FlyEnv::new(self, config)
    }

    pub fn tethered_env(&self, config: TaskConfig) -> Result<TetheredEnv, SimError> {
        TetheredEnv::new(self, config)
    }
}

pub struct FlyWorld {
    data: MjData<Arc<MjModel>>,
    controls: Vec<Control>,
    wing_qpos: Vec<(usize, f64)>,
}

impl FlyWorld {
    pub fn reset(&mut self) {
        self.data.reset();
        for &(address, value) in &self.wing_qpos {
            self.data.qpos_mut()[address] = value;
        }
    }

    /// Apply normalized actions in `[-1, 1]` to the walking actuator set.
    ///
    /// The action order follows flybody's walking environment: claw adhesion,
    /// head, abdomen, then the six legs. Disabled mouth, antenna and wing
    /// actuators are held at zero.
    pub fn step(&mut self, action: &[f32]) -> Result<(), SimError> {
        if action.len() != self.controls.len() {
            return Err(SimError::ActionSize {
                expected: self.controls.len(),
                actual: action.len(),
            });
        }

        for target in self.data.ctrl_mut() {
            *target = 0.0;
        }
        for (control, value) in self.controls.iter().zip(action) {
            let normalized = f64::from(*value).clamp(-1.0, 1.0);
            self.data.ctrl_mut()[control.index] =
                (control.neutral + normalized * control.scale).clamp(control.min, control.max);
        }
        self.data.step();
        Ok(())
    }

    pub fn qpos(&self) -> &[f64] {
        self.data.qpos()
    }

    pub fn qpos_mut(&mut self) -> &mut [f64] {
        self.data.qpos_mut()
    }

    pub fn qvel(&self) -> &[f64] {
        self.data.qvel()
    }

    pub fn qvel_mut(&mut self) -> &mut [f64] {
        self.data.qvel_mut()
    }

    pub fn time(&self) -> f64 {
        self.data.time()
    }

    pub fn forward(&mut self) {
        self.data.forward();
    }

    pub fn body_position(&self, name: &str) -> Option<[f64; 3]> {
        let id = self.data.model().name_to_id(MjtObj::mjOBJ_BODY, name)?;
        self.data.xpos().get(id).copied()
    }

    pub fn joint_qpos_address(&self, name: &str) -> Option<usize> {
        let model = self.data.model();
        let id = model.name_to_id(MjtObj::mjOBJ_JOINT, name)?;
        Some(model.jnt_qposadr()[id] as usize)
    }

    pub fn joint_dof_address(&self, name: &str) -> Option<usize> {
        let model = self.data.model();
        let id = model.name_to_id(MjtObj::mjOBJ_JOINT, name)?;
        Some(model.jnt_dofadr()[id] as usize)
    }

    pub fn contact_count(&self) -> usize {
        self.data.ncon() as usize
    }

    pub fn set_root_pose(&mut self, position: [f64; 3], quaternion: [f64; 4]) {
        let qpos = self.data.qpos_mut();
        if qpos.len() >= 7 {
            qpos[..3].copy_from_slice(&position);
            qpos[3..7].copy_from_slice(&quaternion);
        }
        let qvel = self.data.qvel_mut();
        if qvel.len() >= 6 {
            qvel[..6].fill(0.0);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TaskConfig {
    /// Physics steps per policy action (20 at the flybody 500 Hz control period).
    pub control_decimation: usize,
    /// Maximum number of policy steps in one episode.
    pub episode_control_steps: usize,
    /// Target root height in MuJoCo centimetres. Zero calibrates it from the
    /// lowest claw so the fly starts with its feet on the floor.
    pub target_height: f64,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            control_decimation: 20,
            episode_control_steps: 500,
            target_height: 0.0,
        }
    }
}

pub struct Step {
    pub observation: Vec<f32>,
    pub reward: f64,
    pub done: bool,
}

pub struct FlyEnv {
    world: FlyWorld,
    config: TaskConfig,
    target_height: f64,
    steps: usize,
    last_action: Vec<f32>,
    observation_dim: usize,
}

impl FlyEnv {
    pub fn new(model: &FlyModel, config: TaskConfig) -> Result<Self, SimError> {
        if !model.has_free_root() {
            return Err(SimError::Model {
                path: "model".to_owned(),
                detail: "free-root environment requires a free joint".to_owned(),
            });
        }
        let world = model.world()?;
        let last_action = vec![0.0; model.action_dim()];
        let mut env = Self {
            world,
            config,
            target_height: config.target_height,
            steps: 0,
            last_action,
            observation_dim: model.observation_dim(),
        };
        if env.target_height <= 0.0 {
            env.target_height = env.calibrate_standing_height();
        }
        Ok(env)
    }

    fn calibrate_standing_height(&mut self) -> f64 {
        const NOMINAL: f64 = 0.2;
        self.world.reset();
        self.world
            .set_root_pose([0.0, 0.0, NOMINAL], [1.0, 0.0, 0.0, 0.0]);
        self.world.forward();
        let lowest = ["T1", "T2", "T3"]
            .into_iter()
            .flat_map(|leg| {
                ["left", "right"]
                    .into_iter()
                    .map(move |side| format!("claw_{leg}_{side}"))
            })
            .filter_map(|name| self.world.body_position(&name))
            .map(|position| position[2])
            .fold(f64::INFINITY, f64::min);
        if lowest.is_finite() {
            (NOMINAL - lowest).max(0.01)
        } else {
            NOMINAL
        }
    }

    pub fn reset(&mut self) {
        self.world.reset();
        self.world
            .set_root_pose([0.0, 0.0, self.target_height], [1.0, 0.0, 0.0, 0.0]);
        self.world.forward();
        self.steps = 0;
        self.last_action.fill(0.0);
    }

    pub fn step(&mut self, action: &[f32]) -> Result<Step, SimError> {
        for _ in 0..self.config.control_decimation.max(1) {
            self.world.step(action)?;
        }

        let qpos = self.world.qpos();
        let qvel = self.world.qvel();
        let height = qpos.get(2).copied().unwrap_or_default();
        let (forward_x, forward_y) = forward_xy(qpos);
        let up = up_z(qpos);
        let vx = qvel.first().copied().unwrap_or_default();
        let vy = qvel.get(1).copied().unwrap_or_default();
        let forward_speed = vx * forward_x + vy * forward_y;
        let lateral_speed = -vx * forward_y + vy * forward_x;
        let action_penalty = action
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();

        let mut reward = 0.1 + 2.0 * forward_speed
            - 0.5 * lateral_speed.abs()
            - 1.0 * (height - self.target_height).abs()
            - 1.0 * (1.0 - up).max(0.0)
            - 0.01 * action_penalty;

        self.steps += 1;
        let fell = height < 0.02 || up < 0.2;
        let timed_out = self.steps >= self.config.episode_control_steps;
        if fell {
            reward -= 5.0;
        }

        self.last_action.clear();
        self.last_action.extend_from_slice(action);

        Ok(Step {
            observation: self.observation(),
            reward,
            done: fell || timed_out,
        })
    }

    pub fn observation(&self) -> Vec<f32> {
        let qpos = self.world.qpos();
        let qvel = self.world.qvel();
        let (forward_x, forward_y) = forward_xy(qpos);
        let mut result = Vec::with_capacity(self.observation_dim);

        for index in 0..3 {
            push_scaled(&mut result, qvel.get(index).copied(), 10.0);
        }
        for index in 3..6 {
            push_scaled(&mut result, qvel.get(index).copied(), 0.1);
        }
        push_scaled(&mut result, qpos.get(2).copied(), 10.0);
        push_scaled(&mut result, Some(forward_x), 1.0);
        push_scaled(&mut result, Some(forward_y), 1.0);
        push_scaled(&mut result, Some(up_z(qpos)), 1.0);
        for value in qpos.iter().skip(7) {
            push_scaled(&mut result, Some(*value), 1.0);
        }
        for value in qvel.iter().skip(6) {
            push_scaled(&mut result, Some(*value), 0.1);
        }
        for value in &self.last_action {
            push_scaled(&mut result, Some(f64::from(*value)), 1.0);
        }

        result.resize(self.observation_dim, 0.0);
        result
    }

    pub fn observation_dim(&self) -> usize {
        self.observation_dim
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn target_height(&self) -> f64 {
        self.target_height
    }

    pub fn height(&self) -> f64 {
        self.world.qpos().get(2).copied().unwrap_or_default()
    }

    pub fn forward_speed(&self) -> f64 {
        let qpos = self.world.qpos();
        let qvel = self.world.qvel();
        let (forward_x, forward_y) = forward_xy(qpos);
        let vx = qvel.first().copied().unwrap_or_default();
        let vy = qvel.get(1).copied().unwrap_or_default();
        vx * forward_x + vy * forward_y
    }

    pub fn body_position(&self, name: &str) -> Option<[f64; 3]> {
        self.world.body_position(name)
    }

    pub fn contact_count(&self) -> usize {
        self.world.contact_count()
    }
}

pub struct TetheredEnv {
    world: FlyWorld,
    config: TaskConfig,
    ball_dof: usize,
    action_dim: usize,
    steps: usize,
    last_action: Vec<f32>,
    observation_dim: usize,
}

impl TetheredEnv {
    pub fn new(model: &FlyModel, config: TaskConfig) -> Result<Self, SimError> {
        if model.has_free_root() {
            return Err(SimError::Model {
                path: "model".to_owned(),
                detail: "tethered environment requires a fixed thorax (no free joint)".to_owned(),
            });
        }
        let world = model.world()?;
        let ball_dof = world
            .joint_dof_address("ball")
            .ok_or_else(|| SimError::Model {
                path: "model".to_owned(),
                detail: "tethered model has no ball joint".to_owned(),
            })?;
        let last_action = vec![0.0; model.action_dim()];
        let observation_dim = model.qpos_dim() + model.qvel_dim() + model.action_dim();
        Ok(Self {
            world,
            config,
            ball_dof,
            action_dim: model.action_dim(),
            steps: 0,
            last_action,
            observation_dim,
        })
    }

    pub fn reset(&mut self) {
        self.world.reset();
        if let Some(address) = self.world.joint_qpos_address("ball") {
            let qpos = self.world.qpos_mut();
            qpos[address..address + 4].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        }
        let qvel = self.world.qvel_mut();
        qvel[self.ball_dof..self.ball_dof + 3].fill(0.0);
        self.world.forward();
        self.steps = 0;
        self.last_action.fill(0.0);
    }

    /// Step the tethered fly. The reward tracks the ball's angular velocity
    /// along the flybody walking target `[0, -5, 0]`.
    pub fn step(&mut self, action: &[f32]) -> Result<Step, SimError> {
        for _ in 0..self.config.control_decimation.max(1) {
            self.world.step(action)?;
        }

        let ball_velocity = self.ball_velocity();
        let error = (ball_velocity[0].powi(2)
            + (ball_velocity[1] + 5.0).powi(2)
            + ball_velocity[2].powi(2))
        .sqrt();
        let action_penalty = action
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        let reward = 0.1 + 2.0 * (-error / 5.0).exp() - 0.0001 * action_penalty;

        self.steps += 1;
        self.last_action.clear();
        self.last_action.extend_from_slice(action);
        let done = self.steps >= self.config.episode_control_steps;

        Ok(Step {
            observation: self.observation(),
            reward,
            done,
        })
    }

    pub fn observation(&self) -> Vec<f32> {
        let qpos = self.world.qpos();
        let qvel = self.world.qvel();
        let mut result = Vec::with_capacity(self.observation_dim);

        for value in qpos {
            push_scaled(&mut result, Some(*value), 1.0);
        }
        for value in qvel {
            push_scaled(&mut result, Some(*value), 0.1);
        }
        for value in &self.last_action {
            push_scaled(&mut result, Some(f64::from(*value)), 1.0);
        }

        result.resize(self.observation_dim, 0.0);
        result
    }

    pub fn ball_velocity(&self) -> [f64; 3] {
        let qvel = self.world.qvel();
        [
            qvel.get(self.ball_dof).copied().unwrap_or_default(),
            qvel.get(self.ball_dof + 1).copied().unwrap_or_default(),
            qvel.get(self.ball_dof + 2).copied().unwrap_or_default(),
        ]
    }

    pub fn observation_dim(&self) -> usize {
        self.observation_dim
    }

    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn contact_count(&self) -> usize {
        self.world.contact_count()
    }
}

fn push_scaled(output: &mut Vec<f32>, value: Option<f64>, scale: f64) {
    let value = value.unwrap_or_default() * scale;
    output.push(if value.is_finite() {
        value.clamp(-10.0, 10.0) as f32
    } else {
        0.0
    });
}

fn forward_xy(qpos: &[f64]) -> (f64, f64) {
    if qpos.len() < 7 {
        return (1.0, 0.0);
    }
    let w = qpos[3];
    let x = qpos[4];
    let y = qpos[5];
    let z = qpos[6];
    (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z))
}

fn up_z(qpos: &[f64]) -> f64 {
    if qpos.len() < 7 {
        return 1.0;
    }
    let x = qpos[4];
    let y = qpos[5];
    1.0 - 2.0 * (x * x + y * y)
}

fn prepare_tethered_model(source: &Path, output: &Path) -> Result<(), SimError> {
    let xml = fs::read_to_string(source).map_err(|error| SimError::Model {
        path: source.display().to_string(),
        detail: error.to_string(),
    })?;
    if !xml.contains("<freejoint name=\"free\"/>") {
        if output.exists() {
            return Ok(());
        }
        return Err(SimError::Model {
            path: source.display().to_string(),
            detail: "expected a freejoint named free".to_owned(),
        });
    }

    let mut tethered = xml.replace("<freejoint name=\"free\"/>", "");
    if !tethered.contains("<body name=\"ball\"") {
        let ball = r#"
    <body name="ball" pos="-0.05 0 -0.419">
      <joint name="ball" type="ball"/>
      <geom name="ball" type="sphere" size="0.454" density="0.0025" friction="1 0.005 0.0001" rgba="0.2 0.3 0.4 1"/>
    </body>
"#;
        let index = tethered
            .find("</worldbody>")
            .ok_or_else(|| SimError::Model {
                path: source.display().to_string(),
                detail: "missing worldbody".to_owned(),
            })?;
        tethered.insert_str(index, ball);
    }

    fs::write(output, tethered).map_err(|error| SimError::Model {
        path: output.display().to_string(),
        detail: error.to_string(),
    })?;
    Ok(())
}

fn build_controls(model: &MjModel) -> Vec<Control> {
    let ranges = model.actuator_ctrlrange();
    let transmissions = model.actuator_trnid();
    let transmission_types = model.actuator_trntype();
    let qpos_spring = model.qpos_spring();
    let qpos_addresses = model.jnt_qposadr();
    let mut classes: BTreeMap<u8, Vec<Control>> = BTreeMap::new();

    for (index, range) in ranges.iter().enumerate() {
        let Some(name) = model.id_to_name(MjtObj::mjOBJ_ACTUATOR, index) else {
            continue;
        };
        let class = if name.starts_with("adhere_claw") {
            0
        } else if name == "head" || name.starts_with("head_") {
            1
        } else if name == "abdomen" || name.starts_with("abdomen_") {
            2
        } else if name.contains("_T1_") || name.contains("_T2_") || name.contains("_T3_") {
            3
        } else {
            continue;
        };

        let [min, max] = *range;
        let (min, max) = if max > min { (min, max) } else { (-1.0, 1.0) };
        let neutral = if class == 0 {
            0.0
        } else if transmission_types[index] == MjtTrn::mjTRN_JOINT {
            let joint = transmissions[index][0] as usize;
            let address = qpos_addresses[joint] as usize;
            qpos_spring[address]
        } else {
            0.0
        }
        .clamp(min, max);

        classes.entry(class).or_default().push(Control {
            index,
            min,
            max,
            neutral,
            scale: (max - min) * 0.5,
        });
    }

    classes.into_values().flatten().collect()
}

fn wing_spring_positions(model: &MjModel) -> Vec<(usize, f64)> {
    let mut result = Vec::new();
    let addresses = model.jnt_qposadr();
    let spring = model.qpos_spring();

    for side in ["left", "right"] {
        for axis in ["yaw", "roll", "pitch"] {
            let name = format!("wing_{axis}_{side}");
            let Some(joint) = model.name_to_id(MjtObj::mjOBJ_JOINT, &name) else {
                continue;
            };
            let address = addresses[joint] as usize;
            result.push((address, spring[address]));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const BALL: &str = r#"
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <joint name="ball_joint" type="free"/>
      <geom size=".1"/>
    </body>
    <geom name="floor" type="plane" size="10 10 1"/>
  </worldbody>
  <actuator>
    <motor name="head" joint="ball_joint" gear="1" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"#;

    fn ball_model() -> FlyModel {
        let model = MjModel::from_xml_string(BALL).expect("ball model");
        let controls = build_controls(&model);
        assert_eq!(controls.len(), 1);
        FlyModel {
            model: Arc::new(model),
            timestep: 0.002,
            controls,
            wing_qpos: Vec::new(),
            joint_positions: 0,
            joint_velocities: 0,
            observation_dim: 11,
            has_free_root: true,
        }
    }

    #[test]
    fn world_steps_and_reports_time() {
        let model = ball_model();
        let mut world = model.world().expect("world");
        world.reset();
        world.step(&[0.0]).expect("step");

        assert!((world.time() - 0.002).abs() < 1e-12);
        assert!(!world.qpos().is_empty());
    }

    #[test]
    fn wrong_action_size_is_rejected() {
        let model = ball_model();
        let mut world = model.world().expect("world");

        assert!(matches!(
            world.step(&[]),
            Err(SimError::ActionSize {
                expected: 1,
                actual: 0
            })
        ));
    }

    #[test]
    fn env_step_returns_a_finite_observation_and_reward() {
        let model = ball_model();
        let mut env = model
            .env(TaskConfig {
                control_decimation: 1,
                episode_control_steps: 10,
                target_height: 1.0,
            })
            .expect("env");
        env.reset();
        let step = env.step(&[0.0]).expect("step");

        assert_eq!(step.observation.len(), env.observation_dim());
        assert!(step.reward.is_finite());
    }
}
