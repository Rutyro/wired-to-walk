# Third-party notices

## MaleCNS v1.0 connectome

- Source: <https://male-cns.janelia.org/download/>
- Provider: Janelia Research Campus
- License: CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>)
- Used tables: body annotations, body neurotransmitter predictions and connectome weights
  from the public `flyem-male-cns/v1.0` flat-connectome release.

Derived graph files are generated locally from the public tables and are not
distributed in this repository.

## flybody

- Source: <https://github.com/TuragaLab/flybody>
- Provider: Turaga Lab
- License: Apache-2.0 (the upstream LICENSE is copied with the fetched assets)
- Used assets: the `fruitfly.xml` and `floor.xml` MuJoCo models and their mesh files.

The fetched model is not modified by the fetch script. Any future derived MJCF
will be documented in this file and kept under the upstream license terms.

## Rust dependencies

The dependency list and exact versions are recorded in `Cargo.lock`. Direct
runtime dependencies are licensed under MIT or Apache-2.0; consult the
individual crates for their notices.
