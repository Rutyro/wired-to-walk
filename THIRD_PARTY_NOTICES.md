# Third-party notices

## MaleCNS v1.0 connectome

- Source: <https://male-cns.janelia.org/download/>
- Provider: Janelia Research Campus
- License: CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>)
- Used tables: body annotations, body neurotransmitter predictions and connectome weights
  from the public `flyem-male-cns/v1.0` flat-connectome release.

The tables are downloaded locally by `scripts/fetch-malecns.ps1` and are not
distributed in this repository. Derived graph files are generated locally and
are not distributed either.

## flybody

- Source: <https://github.com/TuragaLab/flybody>
- Provider: Turaga Lab
- License: Apache-2.0 (see `assets/fly/APACHE-2.0.txt`; the upstream LICENSE is
  also copied with the fetched assets)
- Used assets: the `fruitfly.xml` and `floor.xml` MuJoCo models and their mesh files.

The fetched model is not committed. The simulator writes a local derivative,
`data/flybody/wired_tethered.xml`, by removing the free root joint and adding a
floating ball. That derivative is generated at runtime, remains under
Apache-2.0, and is not distributed in this repository.

## MuJoCo

- Source: <https://github.com/google-deepmind/mujoco>
- License: Apache-2.0
- The `mujoco-rs` build downloads the MuJoCo runtime into `data/mujoco/`.
  The runtime is not committed.

## Rust dependencies

The dependency list and exact versions are recorded in `Cargo.lock`. Direct
runtime dependencies are licensed under MIT or Apache-2.0; consult the
individual crates for their notices.
