use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use arrow::array::{Array, Int64Array, StringArray};
use arrow::ipc::reader::FileReader;
use thiserror::Error;

use crate::types::{Connectome, Edge, GraphSource, Leg, Neuron, NeuronRole, Side, Transmitter};

const ANNOTATIONS: &str = "body-annotations-male-cns-v1.0-minconf-0.5.feather";
const TRANSMITTERS: &str = "body-neurotransmitters-male-cns-v1.0.feather";
const WEIGHTS: &str = "connectome-weights-male-cns-v1.0-minconf-0.5.feather";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPaths {
    pub annotations: PathBuf,
    pub transmitters: PathBuf,
    pub weights: PathBuf,
}

impl DataPaths {
    pub fn in_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self {
            annotations: dir.join(ANNOTATIONS),
            transmitters: dir.join(TRANSMITTERS),
            weights: dir.join(WEIGHTS),
        }
    }

    pub fn default_dir() -> Self {
        Self::in_dir("data/malecns")
    }

    pub fn validate(&self) -> Result<(), ConnectomeError> {
        for path in [&self.annotations, &self.transmitters, &self.weights] {
            if !path.is_file() {
                return Err(ConnectomeError::MissingFile(path.clone()));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ConnectomeError {
    #[error("missing data file: {0}")]
    MissingFile(PathBuf),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("missing column {1} in {0}")]
    MissingColumn(String, &'static str),
    #[error("column {1} in {0} has an unexpected type")]
    BadColumn(String, &'static str),
    #[error("duplicate body id {0} in annotations")]
    DuplicateBody(i64),
    #[error("invalid connectome: {0}")]
    Invalid(String),
    #[error("serialization error: {0}")]
    Format(String),
}

pub fn load(paths: &DataPaths) -> Result<Connectome, ConnectomeError> {
    paths.validate()?;

    let transmitters = read_transmitters(&paths.transmitters)?;
    let mut neurons = read_neurons(&paths.annotations, &transmitters)?;
    neurons.sort_by_key(|neuron| neuron.body_id);
    neurons.dedup_by_key(|neuron| neuron.body_id);

    let index: HashMap<i64, u32> = neurons
        .iter()
        .enumerate()
        .map(|(index, neuron)| (neuron.body_id, index as u32))
        .collect();

    let edges = read_edges(&paths.weights, &neurons, &index)?;
    let source = GraphSource {
        annotations: file_name(&paths.annotations),
        transmitters: file_name(&paths.transmitters),
        weights: file_name(&paths.weights),
    };

    Connectome::build(source, neurons, edges).map_err(ConnectomeError::Invalid)
}

pub fn leg_from_nerve(nerve: &str) -> Option<Leg> {
    match nerve {
        "ProLN" => Some(Leg::T1),
        "MesoLN" => Some(Leg::T2),
        "MetaLN" => Some(Leg::T3),
        _ => None,
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_owned()
}

fn read_transmitters(path: &Path) -> Result<HashMap<i64, Transmitter>, ConnectomeError> {
    let name = file_name(path);
    let reader = open_ipc(path)?;
    let mut result = HashMap::new();

    for batch in reader {
        let batch = batch?;
        let bodies = int64_column(&batch, &name, "body")?;
        let labels = string_column(&batch, &name, "consensus_nt")?;

        for row in 0..batch.num_rows() {
            if bodies.is_null(row) {
                continue;
            }
            let label = if labels.is_null(row) {
                None
            } else {
                Some(labels.value(row))
            };
            result.insert(bodies.value(row), Transmitter::from_label(label));
        }
    }

    Ok(result)
}

fn read_neurons(
    path: &Path,
    transmitters: &HashMap<i64, Transmitter>,
) -> Result<Vec<Neuron>, ConnectomeError> {
    let name = file_name(path);
    let reader = open_ipc(path)?;
    let mut neurons = Vec::new();
    let mut seen = HashMap::new();

    for batch in reader {
        let batch = batch?;
        let body_ids = int64_column(&batch, &name, "bodyId")?;
        let instances = string_column(&batch, &name, "instance")?;
        let cell_types = string_column(&batch, &name, "type")?;
        let superclasses = string_column(&batch, &name, "superclass")?;
        let statuses = string_column(&batch, &name, "status")?;
        let soma_sides = string_column(&batch, &name, "somaSide")?;
        let root_sides = string_column(&batch, &name, "rootSide")?;
        let entry_nerves = string_column(&batch, &name, "entryNerve")?;
        let exit_nerves = string_column(&batch, &name, "exitNerve")?;

        for row in 0..batch.num_rows() {
            if body_ids.is_null(row) || statuses.is_null(row) || statuses.value(row) != "Traced" {
                continue;
            }

            let superclass = string_value(superclasses, row).unwrap_or_default();
            let entry_nerve = string_value(entry_nerves, row);
            let exit_nerve = string_value(exit_nerves, row);
            let side = side_from(string_value(soma_sides, row), string_value(root_sides, row));

            let role = match superclass.as_str() {
                "descending_neuron" => Some(NeuronRole::Descending),
                "vnc_intrinsic" => Some(NeuronRole::Intrinsic),
                "vnc_motor" => exit_nerve
                    .as_deref()
                    .and_then(leg_from_nerve)
                    .map(|leg| NeuronRole::Motor { leg, side }),
                "vnc_sensory" => entry_nerve
                    .as_deref()
                    .and_then(leg_from_nerve)
                    .map(|leg| NeuronRole::Sensory { leg, side }),
                _ => None,
            };

            let Some(role) = role else {
                continue;
            };

            let body_id = body_ids.value(row);
            if seen.insert(body_id, ()).is_some() {
                return Err(ConnectomeError::DuplicateBody(body_id));
            }

            let nerve = match role {
                NeuronRole::Motor { .. } => exit_nerve,
                NeuronRole::Sensory { .. } => entry_nerve,
                _ => None,
            };

            neurons.push(Neuron {
                body_id,
                instance: string_value(instances, row).unwrap_or_default(),
                cell_type: string_value(cell_types, row).unwrap_or_default(),
                superclass,
                role,
                nerve,
                side,
                transmitter: transmitters
                    .get(&body_id)
                    .copied()
                    .unwrap_or(Transmitter::Unknown),
            });
        }
    }

    Ok(neurons)
}

fn read_edges(
    path: &Path,
    neurons: &[Neuron],
    index: &HashMap<i64, u32>,
) -> Result<Vec<Edge>, ConnectomeError> {
    let name = file_name(path);
    let reader = open_ipc(path)?;
    let mut edges = Vec::new();

    for batch in reader {
        let batch = batch?;
        let pres = int64_column(&batch, &name, "body_pre")?;
        let posts = int64_column(&batch, &name, "body_post")?;
        let weights = int64_column(&batch, &name, "weight")?;

        for row in 0..batch.num_rows() {
            if pres.is_null(row) || posts.is_null(row) || weights.is_null(row) {
                continue;
            }

            let Some(&pre) = index.get(&pres.value(row)) else {
                continue;
            };
            let Some(&post) = index.get(&posts.value(row)) else {
                continue;
            };

            let sign = neurons[pre as usize].transmitter.sign();
            edges.push(Edge {
                pre,
                post,
                weight: weights.value(row) as f32,
                sign,
            });
        }
    }

    Ok(edges)
}

fn open_ipc(path: &Path) -> Result<FileReader<File>, ConnectomeError> {
    let file = File::open(path)?;
    Ok(FileReader::try_new(file, None)?)
}

fn int64_column<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    file: &str,
    name: &'static str,
) -> Result<&'a Int64Array, ConnectomeError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| ConnectomeError::MissingColumn(file.to_owned(), name))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| ConnectomeError::BadColumn(file.to_owned(), name))
}

fn string_column<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    file: &str,
    name: &'static str,
) -> Result<&'a StringArray, ConnectomeError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| ConnectomeError::MissingColumn(file.to_owned(), name))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ConnectomeError::BadColumn(file.to_owned(), name))
}

fn string_value(column: &StringArray, row: usize) -> Option<String> {
    if column.is_null(row) {
        None
    } else {
        Some(column.value(row).to_owned())
    }
}

fn side_from(soma_side: Option<String>, root_side: Option<String>) -> Side {
    match soma_side
        .as_deref()
        .or(root_side.as_deref())
        .unwrap_or_default()
    {
        "L" | "Left" | "left" => Side::Left,
        "R" | "Right" | "right" => Side::Right,
        _ => Side::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leg_nerves_map_to_thoracic_segments() {
        assert_eq!(leg_from_nerve("ProLN"), Some(Leg::T1));
        assert_eq!(leg_from_nerve("MesoLN"), Some(Leg::T2));
        assert_eq!(leg_from_nerve("MetaLN"), Some(Leg::T3));
        assert_eq!(leg_from_nerve("AbN1"), None);
    }

    #[test]
    fn unknown_side_is_explicit() {
        assert_eq!(side_from(None, Some("L".to_owned())), Side::Left);
        assert_eq!(side_from(Some("M".to_owned()), None), Side::Unknown);
    }
}
