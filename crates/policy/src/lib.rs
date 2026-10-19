use burn::module::Module;
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
}
