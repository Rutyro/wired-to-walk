# wired-to-walk

**A fruit fly learns to walk through a connectome-constrained ventral nerve cord.**

This is a research-engineering portfolio project, not a whole-brain simulation and not a claim that a real fly brain has been uploaded. The policy is a small rate-based recurrent network whose wiring mask and synapse signs come from the MaleCNS v1.0 ventral nerve cord connectome. Only cell-type time constants, biases and connection-type gains are trained. The body and physics come from the openly licensed flybody MuJoCo model.

The project is in **Phase 1: setup and feasibility spike**. The README will only claim results that are reproduced in `results/`.

## Why this exists

Most "fly brain plays X" projects either use a frozen connectome as a readout or train a generic network on a fly body. This project asks a narrower, testable question:

> Can a policy wired like the fly's own ventral nerve cord learn to walk and steer on one consumer GPU, and can it be compared fairly with a same-budget MLP and a degree-preserving shuffled connectome?

The comparison is the point. A negative result is a valid result and will be reported.

## Phase 1 spike

Measured on 2026-10-19 with an RTX 4080 Super 16 GB, Ryzen 5 7600X (6C/12T), 32 GB RAM:

- MaleCNS v1.0 subset: **18,561 neurons** (1,314 descending, 13,151 VNC intrinsic, 319 leg motor, 3,777 leg sensory) and **2,587,861 synapses**.
- flybody walking actuator set: **59 actions** (6 claw adhesion, 3 head, 2 abdomen, 48 leg), **217 observation values**, **108 DOF**.
- Physics: **85,173 steps/s** with 12 parallel worlds on 12 threads, **8.52 simulated seconds per wall second**, **4,259 control steps/s** at the flybody 500 Hz control period. Single-world throughput was 12,235 steps/s.
- Burn 0.22 CUDA forward and backward pass succeeded with CUDA 12.0 (`wired-train smoke`).

The plan's informal gate was about 10,000 control steps/s. The current CPU-physics spike reaches about 4,300 control steps/s, so the gate is **not met**. The fallback is a lighter collision body, a shorter control horizon, or longer training runs. No training result is claimed yet. Raw numbers are in [`results/spike.json`](results/spike.json).

## Status

- [x] Rust workspace and data-fetch scripts
- [x] MaleCNS VNC graph loader (descending, intrinsic, leg motor, leg sensory neurons)
- [x] Flybody MuJoCo spike and throughput measurement
- [x] Burn CUDA smoke test
- [ ] PPO baseline on the fly body
- [ ] Connectome-constrained recurrent policy
- [ ] Custom sparse-step GPU kernel
- [ ] Connectome vs shuffled vs MLP experiment
- [ ] Browser/WASM demo

## Requirements

- Windows 10/11 with Rust stable (MSVC toolchain)
- CUDA 12.x and an NVIDIA GPU for the Burn CUDA backend
- Visual Studio 2022 C++ build tools
- About 2 GB of free disk space for the public connectome tables

The code currently builds with Rust 1.95 or newer. CUDA 12.0 is installed on the development machine; see [Known limitations](#known-limitations).

## Quickstart

Fetch the public data (MaleCNS v1.0 and the flybody MJCF assets):

```powershell
pwsh ./scripts/fetch-malecns.ps1
pwsh ./scripts/fetch-flybody.ps1
```

The `mujoco-rs` build downloads the MuJoCo runtime to
`data/mujoco/mujoco-3.12.0/`. Its `bin` directory must be on `PATH` when tests
or the simulator run:

```powershell
$env:PATH = "$PWD\data\mujoco\mujoco-3.12.0\bin;$env:PATH"
```

Build and run the connectome summary:

```powershell
cargo run --release -p wired-connectome -- build
cargo run --release -p wired-connectome -- summary
```

Run the physics spike:

```powershell
cargo run --release -p wired-sim -- spike --worlds 8 --seconds 10
```

Run the current test suite:

```powershell
cargo test --workspace
```

## Data and licenses

- **MaleCNS v1.0** connectome, Janelia Research Campus, CC BY 4.0. Source: <https://male-cns.janelia.org/download/>. The fetch script downloads the public flat-connectome Feather tables into `data/malecns/`; derived files are not committed.
- **flybody** MuJoCo model, TuragaLab, Apache-2.0. Source: <https://github.com/TuragaLab/flybody>. The fetch script copies the upstream MJCF and mesh assets into `data/flybody/`; the upstream license is copied alongside them.
- The Rust crates used here are listed in `Cargo.lock`; see `THIRD_PARTY_NOTICES.md` for the direct data/model attributions.

## Repository layout

```text
crates/connectome   MaleCNS tables -> typed VNC graph, role/leg mapping, CSR output
crates/sim          flybody MuJoCo environment and parallel-world throughput spike
crates/policy       Burn models: MLP baseline and connectome-constrained RNN
crates/train        training and evaluation binaries
scripts             pinned public-data fetchers
assets/fly          upstream model attribution and usage notes
results             generated metrics and plots
```

## Scientific limitations

- The MaleCNS annotations are predictions; neurotransmitter calls and cell-type labels carry uncertainty. The current sign rule is an explicit heuristic, not a biological ground truth.
- Only descending neurons, ventral-nerve-cord intrinsic neurons, leg motor neurons and leg sensory neurons are used. Brain neurons outside that set are not simulated.
- The connectome gives a wiring mask, not the strength of every synapse. Synapse counts are used as initial weights; trainable per-connection-type gains are the only plasticity in the connectome policy.
- The flybody model is a MuJoCo abstraction. It is not a physical fly, and simulated walking is not evidence of the animal's behaviour.
- No performance claim will be made until the same-budget comparison is complete.

## Known limitations

- CUDA 12.0 is older than some current CUDA 12.x toolchains; if Burn rejects it, the fallback is the Vulkan/wgpu backend or a CUDA toolkit update.
- The browser demo depends on a WASM/WebGPU build of the policy and a MuJoCo WASM adapter; it is deliberately scheduled after the training results.
- The first spikes run physics on the CPU and the policy on the GPU. That split is measured before any long training run.

## License

MIT for this repository's code. Data and model assets retain their upstream licenses; see `THIRD_PARTY_NOTICES.md`.
