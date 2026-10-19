//! MuJoCo fly environment used by the training loop and the throughput spike.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use mujoco_rs::prelude::{MjData, MjModel, MjtObj};
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
}

pub struct FlyModel {
    model: Arc<MjModel>,
    timestep: f64,
    controls: Vec<Control>,
    wing_qpos: Vec<(usize, f64)>,
    observation_dim: usize,
}

impl FlyModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SimError> {
        let path = path.as_ref();
        let model = MjModel::from_xml(path).map_err(|error| SimError::Model {
            path: path.display().to_string(),
            detail: error.to_string(),
        })?;
        let timestep = model.opt().timestep;
        let observation_dim = (model.nq() + model.nv()) as usize;
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

        Ok(Self {
            model: Arc::new(model),
            timestep,
            controls,
            wing_qpos,
            observation_dim,
        })
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

    pub fn body_count(&self) -> usize {
        self.model.nbody() as usize
    }

    pub fn dof_count(&self) -> usize {
        self.model.nv() as usize
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
            let span = control.max - control.min;
            self.data.ctrl_mut()[control.index] = control.min + (normalized + 1.0) * 0.5 * span;
        }
        self.data.step();
        Ok(())
    }

    pub fn qpos(&self) -> &[f64] {
        self.data.qpos()
    }

    pub fn qvel(&self) -> &[f64] {
        self.data.qvel()
    }

    pub fn time(&self) -> f64 {
        self.data.time()
    }
}

fn build_controls(model: &MjModel) -> Vec<Control> {
    let ranges = model.actuator_ctrlrange();
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
        classes
            .entry(class)
            .or_default()
            .push(Control { index, min, max });
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
            observation_dim: 13,
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
}
