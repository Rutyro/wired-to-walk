pub mod io;
pub mod male_cns;
pub mod summary;
pub mod types;

pub use male_cns::{ConnectomeError, DataPaths, load};
pub use summary::GraphSummary;
pub use types::{Connectome, Edge, GraphSource, Leg, Neuron, NeuronRole, Side, Transmitter};

#[cfg(test)]
mod tests {
    use super::*;

    fn neuron(body_id: i64, role: NeuronRole, transmitter: Transmitter) -> Neuron {
        Neuron {
            body_id,
            instance: format!("n{body_id}"),
            cell_type: "test".to_owned(),
            superclass: role.label().to_owned(),
            role,
            nerve: None,
            side: Side::Left,
            transmitter,
        }
    }

    #[test]
    fn csr_groups_edges_by_presynaptic_index() {
        let neurons = vec![
            neuron(1, NeuronRole::Descending, Transmitter::Acetylcholine),
            neuron(2, NeuronRole::Intrinsic, Transmitter::Gaba),
            neuron(
                3,
                NeuronRole::Motor {
                    leg: Leg::T1,
                    side: Side::Left,
                },
                Transmitter::Glutamate,
            ),
        ];
        let edges = vec![
            Edge {
                pre: 0,
                post: 1,
                weight: 3.0,
                sign: 1.0,
            },
            Edge {
                pre: 0,
                post: 2,
                weight: 2.0,
                sign: 1.0,
            },
            Edge {
                pre: 1,
                post: 2,
                weight: 1.0,
                sign: -1.0,
            },
        ];

        let graph = Connectome::build(
            GraphSource {
                annotations: "a".to_owned(),
                transmitters: "t".to_owned(),
                weights: "w".to_owned(),
            },
            neurons,
            edges,
        )
        .expect("graph should be valid");

        assert_eq!(graph.outgoing(0).len(), 2);
        assert_eq!(graph.outgoing(1).len(), 1);
        assert!(graph.outgoing(2).is_empty());
        assert_eq!(graph.outgoing(1)[0].weight, 1.0);
    }

    #[test]
    fn invalid_edge_is_rejected() {
        let neurons = vec![neuron(1, NeuronRole::Intrinsic, Transmitter::Unknown)];
        let edges = vec![Edge {
            pre: 0,
            post: 1,
            weight: 1.0,
            sign: 1.0,
        }];

        let result = Connectome::build(
            GraphSource {
                annotations: "a".to_owned(),
                transmitters: "t".to_owned(),
                weights: "w".to_owned(),
            },
            neurons,
            edges,
        );

        assert!(result.is_err());
    }

    #[test]
    fn graph_round_trips_through_bincode() {
        let graph = Connectome::build(
            GraphSource {
                annotations: "a".to_owned(),
                transmitters: "t".to_owned(),
                weights: "w".to_owned(),
            },
            vec![neuron(1, NeuronRole::Intrinsic, Transmitter::Unknown)],
            Vec::new(),
        )
        .expect("graph should be valid");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("graph.bin");

        io::save(&graph, &path).expect("save");
        let loaded = io::load(&path).expect("load");

        assert_eq!(graph, loaded);
    }
}
