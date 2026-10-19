use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

/// A same-budget MLP baseline for the fly body.
#[derive(Module, Debug)]
pub struct MlpPolicy {
    input: Linear,
    hidden: Linear,
    output: Linear,
}

impl MlpPolicy {
    pub fn new(input_dim: usize, hidden_dim: usize, action_dim: usize, device: &Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, hidden_dim).init(device),
            hidden: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            output: LinearConfig::new(hidden_dim, action_dim).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let hidden = Relu::new().forward(self.input.forward(input));
        let hidden = Relu::new().forward(self.hidden.forward(hidden));
        self.output.forward(hidden)
    }

    pub fn parameter_count(&self) -> usize {
        count(&self.input.weight.val())
            + self
                .input
                .bias
                .as_ref()
                .map_or(0, |bias| count(&bias.val()))
            + count(&self.hidden.weight.val())
            + self
                .hidden
                .bias
                .as_ref()
                .map_or(0, |bias| count(&bias.val()))
            + count(&self.output.weight.val())
            + self
                .output
                .bias
                .as_ref()
                .map_or(0, |bias| count(&bias.val()))
    }
}

fn count<const D: usize>(tensor: &Tensor<D>) -> usize {
    tensor.dims().iter().product()
}

/// Gaussian actor with a linear value head used by PPO.
#[derive(Module, Debug)]
pub struct ActorCritic {
    trunk1: Linear,
    trunk2: Linear,
    actor: Linear,
    critic: Linear,
    log_std: Param<Tensor<2>>,
}

impl ActorCritic {
    pub fn new(input_dim: usize, hidden_dim: usize, action_dim: usize, device: &Device) -> Self {
        Self {
            trunk1: LinearConfig::new(input_dim, hidden_dim).init(device),
            trunk2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            actor: LinearConfig::new(hidden_dim, action_dim).init(device),
            critic: LinearConfig::new(hidden_dim, 1).init(device),
            log_std: Param::from_tensor(Tensor::<2>::full([1, action_dim], -0.5, device)),
        }
    }

    pub fn forward(&self, observation: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let hidden = Relu::new().forward(self.trunk1.forward(observation));
        let hidden = Relu::new().forward(self.trunk2.forward(hidden));
        (
            self.actor.forward(hidden.clone()),
            self.critic.forward(hidden),
        )
    }

    pub fn log_std(&self) -> Tensor<2> {
        self.log_std.val()
    }

    pub fn parameter_count(&self) -> usize {
        self.num_params()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_has_expected_output_shape() {
        let device = Device::flex();
        let policy = MlpPolicy::new(12, 64, 18, &device);
        let input = Tensor::<2>::zeros([4, 12], &device);
        let output = policy.forward(input);

        assert_eq!(output.dims(), [4, 18]);
    }

    #[test]
    fn actor_critic_has_expected_output_shapes() {
        let device = Device::flex();
        let model = ActorCritic::new(12, 64, 18, &device);
        let input = Tensor::<2>::zeros([4, 12], &device);
        let (mean, value) = model.forward(input);

        assert_eq!(mean.dims(), [4, 18]);
        assert_eq!(value.dims(), [4, 1]);
        assert_eq!(model.log_std().dims(), [1, 18]);
    }
}
