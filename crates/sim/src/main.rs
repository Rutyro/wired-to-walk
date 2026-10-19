use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;
use wired_sim::FlyModel;

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
         spike [--model FILE] [--worlds N] [--threads N] [--seconds N] [--results FILE]\n\n\
         Defaults: data/flybody/fruitfly.xml, 8 worlds, all logical CPUs, 10 seconds,\n\
         results/spike.json. Controls are held at zero; this measures physics throughput."
    );
}
