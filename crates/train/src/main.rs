use std::env;
use std::error::Error;

use burn::prelude::*;
use wired_policy::MlpPolicy;

const OBSERVATION_DIM: usize = 217;
const ACTION_DIM: usize = 59;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "smoke".to_owned());

    match command.as_str() {
        "smoke" => {
            let use_cpu = args.any(|arg| arg == "--cpu");
            let device = if use_cpu {
                Device::flex()
            } else {
                Device::cuda(0)
            }
            .autodiff();

            let policy = MlpPolicy::new(OBSERVATION_DIM, 256, ACTION_DIM, &device);
            let input = Tensor::<2>::ones([8, OBSERVATION_DIM], &device).require_grad();
            let output = policy.forward(input.clone());
            let loss = output.clone().sum();
            let gradients = loss.backward();
            let input_gradient = input.grad(&gradients).expect("input gradient");

            println!("device: {device:?}");
            println!("parameters: {}", policy.parameter_count());
            println!("input: {:?}", input.dims());
            println!("output: {:?}", output.dims());
            println!("input gradient: {:?}", input_gradient.dims());
        }
        "help" | "--help" | "-h" => print_help(),
        other => return Err(format!("unknown command: {other}").into()),
    }

    Ok(())
}

fn print_help() {
    println!(
        "wired-train\n\n\
         smoke [--cpu]  initialize the policy, run a forward pass and one backward pass\n\n\
         The default device is CUDA 0 with autodiff enabled. --cpu uses the Flex backend."
    );
}
