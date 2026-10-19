use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Left,
    Right,
    Unknown,
}

impl Side {
    pub fn label(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Leg {
    T1,
    T2,
    T3,
}

impl Leg {
    pub const ALL: [Self; 3] = [Self::T1, Self::T2, Self::T3];

    pub fn label(self) -> &'static str {
        match self {
            Self::T1 => "T1",
            Self::T2 => "T2",
            Self::T3 => "T3",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronRole {
    Descending,
    Intrinsic,
    Motor { leg: Leg, side: Side },
    Sensory { leg: Leg, side: Side },
}

impl NeuronRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::Descending => "descending",
            Self::Intrinsic => "intrinsic",
            Self::Motor { .. } => "motor",
            Self::Sensory { .. } => "sensory",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transmitter {
    Acetylcholine,
    Glutamate,
    Gaba,
    Histamine,
    Dopamine,
    Octopamine,
    Serotonin,
    Unclear,
    Unknown,
}

impl Transmitter {
    pub fn from_label(label: Option<&str>) -> Self {
        match label.unwrap_or_default().to_ascii_lowercase().as_str() {
            "acetylcholine" => Self::Acetylcholine,
            "glutamate" => Self::Glutamate,
            "gaba" => Self::Gaba,
            "histamine" => Self::Histamine,
            "dopamine" => Self::Dopamine,
            "octopamine" => Self::Octopamine,
            "serotonin" => Self::Serotonin,
            "unclear" => Self::Unclear,
            _ => Self::Unknown,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Acetylcholine => "acetylcholine",
            Self::Glutamate => "glutamate",
            Self::Gaba => "gaba",
            Self::Histamine => "histamine",
            Self::Dopamine => "dopamine",
            Self::Octopamine => "octopamine",
            Self::Serotonin => "serotonin",
            Self::Unclear => "unclear",
            Self::Unknown => "unknown",
        }
    }

    /// Sign heuristic for the first model.
    ///
    /// GABA and histamine are treated as inhibitory. Other transmitters are
    /// excitatory; unclear calls default to excitation. This is a documented
    /// heuristic, not a receptor-level ground truth.
    pub fn sign(self) -> f32 {
        match self {
            Self::Gaba | Self::Histamine => -1.0,
            _ => 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Neuron {
    pub body_id: i64,
    pub instance: String,
    pub cell_type: String,
    pub superclass: String,
    pub role: NeuronRole,
    pub nerve: Option<String>,
    pub side: Side,
    pub transmitter: Transmitter,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub pre: u32,
    pub post: u32,
    pub weight: f32,
    pub sign: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphSource {
    pub annotations: String,
    pub transmitters: String,
    pub weights: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Connectome {
    pub source: GraphSource,
    pub neurons: Vec<Neuron>,
    pub edges: Vec<Edge>,
    /// CSR offsets over `edges`, which are sorted by presynaptic index.
    pub outgoing_offsets: Vec<u32>,
}

impl Connectome {
    pub fn build(
        source: GraphSource,
        neurons: Vec<Neuron>,
        mut edges: Vec<Edge>,
    ) -> Result<Self, String> {
        let n = neurons.len();
        if n > u32::MAX as usize {
            return Err("neuron count exceeds u32 index space".to_owned());
        }

        edges.sort_by_key(|edge| (edge.pre, edge.post));
        let mut outgoing_offsets = vec![0_u32; n + 1];
        for edge in &edges {
            let pre = edge.pre as usize;
            if pre >= n {
                return Err(format!("edge presynaptic index {pre} is out of range"));
            }
            outgoing_offsets[pre + 1] += 1;
        }
        for index in 1..outgoing_offsets.len() {
            outgoing_offsets[index] += outgoing_offsets[index - 1];
        }

        let graph = Self {
            source,
            neurons,
            edges,
            outgoing_offsets,
        };
        graph.validate()?;
        Ok(graph)
    }

    pub fn outgoing(&self, neuron: usize) -> &[Edge] {
        let start = self.outgoing_offsets[neuron] as usize;
        let end = self.outgoing_offsets[neuron + 1] as usize;
        &self.edges[start..end]
    }

    pub fn neuron_index(&self) -> std::collections::HashMap<i64, u32> {
        self.neurons
            .iter()
            .enumerate()
            .map(|(index, neuron)| (neuron.body_id, index as u32))
            .collect()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.outgoing_offsets.len() != self.neurons.len() + 1 {
            return Err("CSR offset length does not match neuron count".to_owned());
        }
        if self.outgoing_offsets[0] != 0 {
            return Err("CSR offsets must start at zero".to_owned());
        }
        if self.outgoing_offsets.last().copied() != Some(self.edges.len() as u32) {
            return Err("CSR offsets do not cover every edge".to_owned());
        }

        let n = self.neurons.len() as u32;
        let mut previous = None;
        for edge in &self.edges {
            if edge.pre >= n || edge.post >= n {
                return Err("edge index is out of range".to_owned());
            }
            let key = (edge.pre, edge.post);
            if previous.is_some_and(|value| value > key) {
                return Err("edges are not sorted by presynaptic index".to_owned());
            }
            previous = Some(key);
            if !edge.weight.is_finite() || edge.weight < 0.0 {
                return Err("edge weight must be finite and non-negative".to_owned());
            }
            if !edge.sign.is_finite() {
                return Err("edge sign must be finite".to_owned());
            }
        }

        for window in self.outgoing_offsets.windows(2) {
            if window[0] > window[1] {
                return Err("CSR offsets must be monotonic".to_owned());
            }
        }

        Ok(())
    }
}
