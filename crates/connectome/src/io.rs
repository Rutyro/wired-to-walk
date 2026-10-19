use std::fs;
use std::path::Path;

use crate::male_cns::ConnectomeError;
use crate::types::Connectome;

pub fn save(graph: &Connectome, path: impl AsRef<Path>) -> Result<(), ConnectomeError> {
    let bytes = bincode::serde::encode_to_vec(graph, bincode::config::standard())
        .map_err(|error| ConnectomeError::Format(error.to_string()))?;
    fs::write(path, bytes)?;
    Ok(())
}

pub fn load(path: impl AsRef<Path>) -> Result<Connectome, ConnectomeError> {
    let bytes = fs::read(path)?;
    let (graph, _): (Connectome, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .map_err(|error| ConnectomeError::Format(error.to_string()))?;
    graph.validate().map_err(ConnectomeError::Invalid)?;
    Ok(graph)
}
