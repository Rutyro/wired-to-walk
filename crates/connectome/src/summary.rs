use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::types::{Connectome, NeuronRole};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphSummary {
    pub source: crate::types::GraphSource,
    pub neurons: usize,
    pub edges: usize,
    pub roles: BTreeMap<String, usize>,
    pub transmitters: BTreeMap<String, usize>,
    pub inhibitory_edges: usize,
    pub excitatory_edges: usize,
    pub total_synapses: f64,
    pub mean_weight: f64,
}

impl GraphSummary {
    pub fn from_graph(graph: &Connectome) -> Self {
        let mut roles = BTreeMap::new();
        let mut transmitters = BTreeMap::new();

        for neuron in &graph.neurons {
            let role = match neuron.role {
                NeuronRole::Descending => "descending".to_owned(),
                NeuronRole::Intrinsic => "intrinsic".to_owned(),
                NeuronRole::Motor { leg, side } => {
                    format!("motor:{}:{}", leg.label(), side.label())
                }
                NeuronRole::Sensory { leg, side } => {
                    format!("sensory:{}:{}", leg.label(), side.label())
                }
            };
            *roles.entry(role).or_insert(0) += 1;
            *transmitters
                .entry(neuron.transmitter.label().to_owned())
                .or_insert(0) += 1;
        }

        let mut inhibitory_edges = 0;
        let mut total_synapses = 0.0_f64;
        for edge in &graph.edges {
            if edge.sign < 0.0 {
                inhibitory_edges += 1;
            }
            total_synapses += f64::from(edge.weight);
        }

        let edges = graph.edges.len();
        let mean_weight = if edges == 0 {
            0.0
        } else {
            total_synapses / edges as f64
        };

        Self {
            source: graph.source.clone(),
            neurons: graph.neurons.len(),
            edges,
            roles,
            transmitters,
            excitatory_edges: edges - inhibitory_edges,
            inhibitory_edges,
            total_synapses,
            mean_weight,
        }
    }
}
