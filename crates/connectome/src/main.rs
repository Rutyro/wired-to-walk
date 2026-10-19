use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use wired_connectome::{DataPaths, GraphSummary, io, load};

const DEFAULT_DATA: &str = "data/malecns";
const DEFAULT_GRAPH: &str = "data/connectome.bin";

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
        "build" => {
            let mut data = PathBuf::from(DEFAULT_DATA);
            let mut output = PathBuf::from(DEFAULT_GRAPH);
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--data" => data = PathBuf::from(require_value(&mut args, "--data")?),
                    "--output" => output = PathBuf::from(require_value(&mut args, "--output")?),
                    _ => return Err(format!("unknown argument: {flag}").into()),
                }
            }

            let paths = DataPaths::in_dir(&data);
            let started = Instant::now();
            let graph = load(&paths)?;
            let summary = GraphSummary::from_graph(&graph);
            io::save(&graph, &output)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
            eprintln!(
                "built {} neurons and {} edges in {:.2?}; wrote {}",
                summary.neurons,
                summary.edges,
                started.elapsed(),
                output.display()
            );
        }
        "summary" => {
            let mut graph_path = PathBuf::from(DEFAULT_GRAPH);
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--graph" => graph_path = PathBuf::from(require_value(&mut args, "--graph")?),
                    _ => return Err(format!("unknown argument: {flag}").into()),
                }
            }

            let graph = io::load(&graph_path)?;
            let summary = GraphSummary::from_graph(&graph);
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        "validate" => {
            let mut graph_path = PathBuf::from(DEFAULT_GRAPH);
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--graph" => graph_path = PathBuf::from(require_value(&mut args, "--graph")?),
                    _ => return Err(format!("unknown argument: {flag}").into()),
                }
            }

            let graph = io::load(&graph_path)?;
            graph.validate()?;
            println!(
                "ok: {} neurons, {} edges",
                graph.neurons.len(),
                graph.edges.len()
            );
        }
        "help" | "--help" | "-h" => print_help(),
        other => return Err(format!("unknown command: {other}").into()),
    }

    Ok(())
}

fn require_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn print_help() {
    println!(
        "wired-connectome\n\n\
         Commands:\n\
           build   [--data DIR] [--output FILE]  build data/connectome.bin from MaleCNS\n\
           summary [--graph FILE]                print graph counts as JSON\n\
           validate [--graph FILE]               validate the serialized graph\n\n\
         Defaults: data/malecns -> data/connectome.bin"
    );
}
