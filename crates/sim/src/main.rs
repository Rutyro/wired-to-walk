use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;
use wired_sim::{FlyModel, TaskConfig};

const DEFAULT_MODEL: &str = "data/flybody/fruitfly.xml";
const DEFAULT_RESULTS: &str = "results/spike.json";
const CONTROL_PERIOD_SECONDS: f64 = 0.002;

#[derive(Debug, Serialize)]
struct SpikeResult {
    model: String,
    worlds: usize,
    threads: usize,
    requested_seconds: f64,
    timestep: f64,
    steps_per_world: usize,
    total_steps: u64,
    elapsed_seconds: f64,
    steps_per_second: f64,
    simulated_seconds_per_wall_second: f64,
    control_period_seconds: f64,
    control_steps_per_second: f64,
    single_world_steps_per_second: f64,
    action_dim: usize,
    observation_dim: usize,
    body_count: usize,
    dof_count: usize,
    control: &'static str,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut model_path = PathBuf::from(DEFAULT_MODEL);
    let mut results_path = PathBuf::from(DEFAULT_RESULTS);
    let mut worlds = 8_usize;
    let mut threads = std::thread::available_parallelism()?.get();
    let mut seconds = 10.0_f64;

    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "spike".to_owned());
    if command == "help" || command == "--help" || command == "-h" {
        print_help();
        return Ok(());
    }
    if command == "probe" {
        let mut tethered = false;
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--model" => model_path = PathBuf::from(value(&mut args, "--model")?),
                "--seconds" => seconds = value(&mut args, "--seconds")?.parse()?,
                "--tethered" => tethered = true,
                _ => return Err(format!("unknown argument: {flag}").into()),
            }
        }
        return if tethered {
            probe_tethered(&model_path, seconds)
        } else {
            probe(&model_path, seconds)
        };
    }
    if command != "spike" {
        return Err(format!("unknown command: {command}").into());
    }

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => model_path = PathBuf::from(value(&mut args, "--model")?),
            "--results" => results_path = PathBuf::from(value(&mut args, "--results")?),
            "--worlds" => worlds = value(&mut args, "--worlds")?.parse()?,
            "--threads" => threads = value(&mut args, "--threads")?.parse()?,
            "--seconds" => seconds = value(&mut args, "--seconds")?.parse()?,
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }
    if worlds == 0 || threads == 0 || seconds <= 0.0 {
        return Err("worlds, threads and seconds must be positive".into());
    }

    let model = FlyModel::load(&model_path)?;
    let steps_per_world = (seconds / model.timestep()).round().max(1.0) as usize;
    let action = vec![0.0_f32; model.action_dim()];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;

    let single_seconds = seconds.min(2.0);
    let single_steps = (single_seconds / model.timestep()).round().max(1.0) as usize;
    let single_started = Instant::now();
    let mut single_world = model.world()?;
    single_world.reset();
    for _ in 0..single_steps {
        single_world.step(&action)?;
    }
    let single_elapsed = single_started.elapsed().as_secs_f64();
    let single_world_sps = single_steps as f64 / single_elapsed.max(f64::EPSILON);

    let started = Instant::now();
    let counts: Result<Vec<u64>, String> = pool.install(|| {
        (0..worlds)
            .into_par_iter()
            .map(|_| {
                let mut world = model.world().map_err(|error| error.to_string())?;
                world.reset();
                for _ in 0..steps_per_world {
                    world.step(&action).map_err(|error| error.to_string())?;
                }
                Ok(steps_per_world as u64)
            })
            .collect()
    });
    let total_steps: u64 = counts?.into_iter().sum();
    let elapsed = started.elapsed().as_secs_f64();
    let steps_per_second = total_steps as f64 / elapsed.max(f64::EPSILON);
    let simulated_seconds_per_wall_second = steps_per_second * model.timestep();

    let result = SpikeResult {
        model: display_path(&model_path),
        worlds,
        threads,
        requested_seconds: seconds,
        timestep: model.timestep(),
        steps_per_world,
        total_steps,
        elapsed_seconds: elapsed,
        steps_per_second,
        simulated_seconds_per_wall_second,
        control_period_seconds: CONTROL_PERIOD_SECONDS,
        control_steps_per_second: simulated_seconds_per_wall_second / CONTROL_PERIOD_SECONDS,
        single_world_steps_per_second: single_world_sps,
        action_dim: model.action_dim(),
        observation_dim: model.observation_dim(),
        body_count: model.body_count(),
        dof_count: model.dof_count(),
        control: "neutral normalized actions",
    };

    if let Some(parent) = results_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&results_path, serde_json::to_vec_pretty(&result)?)?;
    println!("{}", serde_json::to_string_pretty(&result)?);

    Ok(())
}

fn probe(model_path: &Path, seconds: f64) -> Result<(), Box<dyn Error>> {
    let model = FlyModel::load(model_path)?;
    let config = TaskConfig::default();
    let mut env = model.env(config)?;
    env.reset();
    let steps = (seconds / (model.timestep() * config.control_decimation as f64))
        .round()
        .max(1.0) as usize;
    let action = vec![0.0_f32; model.action_dim()];

    println!(
        "model={} action_dim={} observation_dim={} control_decimation={} steps={steps}",
        model_path.display(),
        model.action_dim(),
        env.observation_dim(),
        config.control_decimation
    );
    println!(
        "reset: height={:.4} contacts={}",
        env.height(),
        env.contact_count()
    );
    for leg in ["T1", "T2", "T3"] {
        for side in ["left", "right"] {
            let name = format!("claw_{leg}_{side}");
            if let Some(position) = env.body_position(&name) {
                println!(
                    "  {name}: x={:.4} y={:.4} z={:.4}",
                    position[0], position[1], position[2]
                );
            }
        }
    }

    let mut total_reward = 0.0;
    let mut done_at = None;
    for step in 0..steps {
        let result = env.step(&action)?;
        total_reward += result.reward;
        if step % 20 == 0 || result.done {
            println!(
                "step={} height={:.4} contacts={} forward_speed={:.4} reward={:.4} done={}",
                step + 1,
                env.height(),
                env.contact_count(),
                env.forward_speed(),
                result.reward,
                result.done
            );
        }
        if result.done {
            done_at = Some(step + 1);
            break;
        }
    }
    println!(
        "summary: steps={} mean_reward={:.4} done_at={done_at:?}",
        done_at.unwrap_or(steps),
        total_reward / done_at.unwrap_or(steps) as f64
    );

    Ok(())
}

fn probe_tethered(model_path: &Path, seconds: f64) -> Result<(), Box<dyn Error>> {
    let model = FlyModel::load_tethered(model_path)?;
    let config = TaskConfig::default();
    let mut env = model.tethered_env(config)?;
    env.reset();
    let steps = (seconds / (model.timestep() * config.control_decimation as f64))
        .round()
        .max(1.0) as usize;
    let action = vec![0.0_f32; model.action_dim()];

    println!(
        "tethered model={} action_dim={} observation_dim={} control_decimation={} steps={steps}",
        model_path.display(),
        model.action_dim(),
        env.observation_dim(),
        config.control_decimation
    );
    println!(
        "reset: ball_qvel={:?} contacts={}",
        env.ball_velocity(),
        env.contact_count()
    );

    let mut total_reward = 0.0;
    for step in 0..steps {
        let result = env.step(&action)?;
        total_reward += result.reward;
        if step % 20 == 0 || result.done {
            println!(
                "step={} ball_qvel={:?} contacts={} reward={:.4} done={}",
                step + 1,
                env.ball_velocity(),
                env.contact_count(),
                result.reward,
                result.done
            );
        }
        if result.done {
            break;
        }
    }
    println!(
        "summary: steps={} mean_reward={:.4}",
        env.steps(),
        total_reward / env.steps().max(1) as f64
    );

    Ok(())
}

fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn print_help() {
    println!(
        "wired-sim\n\n\
         spike [--model FILE] [--worlds N] [--threads N] [--seconds N] [--results FILE]\n\
         probe [--model FILE] [--seconds N] [--tethered]\n\n\
         Defaults: data/flybody/fruitfly.xml, 8 worlds, all logical CPUs, 10 seconds,\n\
         results/spike.json. Controls are held at neutral normalized values.\n\
         probe prints the reset height and a short neutral-action rollout."
    );
}
