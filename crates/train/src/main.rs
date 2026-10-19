mod ppo;

use std::env;
use std::error::Error;
use std::path::PathBuf;

use burn::prelude::*;
use wired_policy::MlpPolicy;

const OBSERVATION_DIM: usize = 273;
const ACTION_DIM: usize = 59;
const DEFAULT_MODEL: &str = "data/flybody/fruitfly.xml";

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "help".to_owned());

    match command.as_str() {
        "smoke" => smoke(args),
        "ppo" => run_ppo(args),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown command: {other}").into()),
    }
}

fn smoke(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn Error>> {
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
    Ok(())
}

fn run_ppo(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn Error>> {
    let mut config = ppo::PpoConfig::default();
    let mut model_path = PathBuf::from(DEFAULT_MODEL);
    let mut output_path = ppo::default_output();
    let mut cpu = false;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--updates" => config.updates = value(&mut args, "--updates")?.parse()?,
            "--envs" => config.envs = value(&mut args, "--envs")?.parse()?,
            "--rollout" => config.rollout_steps = value(&mut args, "--rollout")?.parse()?,
            "--epochs" => config.epochs = value(&mut args, "--epochs")?.parse()?,
            "--minibatch" => config.minibatch = value(&mut args, "--minibatch")?.parse()?,
            "--lr" => config.learning_rate = value(&mut args, "--lr")?.parse()?,
            "--seed" => config.seed = value(&mut args, "--seed")?.parse()?,
            "--episode-steps" => {
                config.episode_steps = value(&mut args, "--episode-steps")?.parse()?
            }
            "--model" => model_path = PathBuf::from(value(&mut args, "--model")?),
            "--output" => output_path = PathBuf::from(value(&mut args, "--output")?),
            "--cpu" => cpu = true,
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }

    if config.envs == 0
        || config.rollout_steps == 0
        || config.epochs == 0
        || config.minibatch == 0
        || config.episode_steps == 0
    {
        return Err("envs, rollout, epochs, minibatch and episode-steps must be positive".into());
    }

    ppo::run(&model_path, &output_path, config, cpu)
}

fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn print_help() {
    println!(
        "wired-train\n\n\
         smoke [--cpu]\n\
           initialize the MLP, run a forward pass and one backward pass\n\n\
         ppo [options]\n\
           train the MLP baseline on the tethered fly-on-ball task\n\n\
           --updates N        PPO updates (default 100)\n\
           --envs N           parallel environments (default 8)\n\
           --rollout N        steps per environment per update (default 64)\n\
           --epochs N         optimization epochs per update (default 4)\n\
           --minibatch N      minibatch size (default 256)\n\
           --lr F             learning rate (default 3e-4)\n\
           --seed N           random seed (default 7)\n\
           --episode-steps N  policy steps per episode (default 500)\n\
           --model FILE       flybody XML (default data/flybody/fruitfly.xml)\n\
           --output FILE      report path (default results/ppo.json)\n\
           --cpu              use the Flex CPU backend\n\n\
         The default device is CUDA 0 with autodiff enabled."
    );
}
