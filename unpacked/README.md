# PhDN: Physics-Embedded Dictionary Network

This repository contains the MATLAB implementation and experiment drivers used in the PhDN study. The main framework is implemented in MATLAB, while several comparison methods—including PySR-based symbolic regression (SR), KAN, MLP, and EQL-Div—are executed through Python adapters called from MATLAB.

> **Large experiment data and trained models are distributed separately.**  
> Download the supplementary data package from: **[\[DATA_DOWNLOAD_LINK\]](https://drive.google.com/file/d/1rXuBUDSU4PxXsUlEuBx9bF2gf5TabFxR/view?usp=sharing)**  **[\[\\[DATA_DOWNLOAD_LINK\\]\]](https://drive.google.com/file/d/1uoIL0bR3r9Y7qatRmm4FYaplonLZV_OR/view?usp=sharing)**
> After downloading, place the supplied `outputs/` and `result for paper/` folders directly under the repository root, as described below.

---

## 1. Environment Setup

### 1.1 MATLAB

The main experiment interface is MATLAB Live Scripts (`.mlx`). Set the repository root as the MATLAB **Current Folder** before running any experiment.

The code automatically adds the main source folders to the MATLAB path. The repository root should contain folders such as:

```text
baselines/
core/
system_identification/
tasks/
third_party/
tools/
utils/
```

Recommended MATLAB version used for the paper:

```text
[MATLAB_VERSION]
```

If a different MATLAB release is used, minor compatibility differences may occur.

### 1.2 Python

Python is required by several baseline adapters:

- **PySR / SR**: symbolic-regression search used in PhDN Stage 0 and the SR baseline.
- **KAN**: based on the bundled `third_party/pykan/` implementation.
- **MLP**: uses the PyTorch/pyKAN numerical infrastructure.
- **EQL-Div**: uses its own legacy Python/Theano environment.

The MATLAB demo files contain explicit Python executable settings. Before running any training experiment, update these paths in the corresponding **root-level experiment Live Script (`.mlx`)** to match your local installation.

For the two system-identification experiments, edit:

```text
run_demo_single_generator_dynamic.mlx
run_demo_soft_saturated_lorenz96.mlx
```

For the other experiments, edit the corresponding root-level `run_demo_*.mlx` file.

Inside each Live Script, use MATLAB's search function to locate:

```matlab
Stage0PythonExe = 'C:\path\to\your\python.exe';
```

`Stage0PythonExe` is used by the PySR-based Stage-0 symbolic-regression search and is also reused by several Python-based baselines, including KAN and MLP.

If EQL-Div is enabled, also search for:

```matlab
EQLPythonExe = 'C:\path\to\your\eql_python.exe';
```

and replace it with the Python executable of the dedicated EQL environment.

For example, in

```text
run_demo_single_generator_dynamic.mlx
```

the relevant variables are defined in the initial configuration sections for Stage-0/Python execution and the EQL baseline. The same variables are defined in

```text
run_demo_soft_saturated_lorenz96.mlx
```

for the Lorenz--96 experiment.

A convenient way to locate these settings is to search directly for `Stage0PythonExe` and `EQLPythonExe` inside the Live Script.

Do **not** leave the author-machine paths unchanged unless they exist on your system.

#### PySR / KAN / MLP environment

The Stage-0 PySR interface requires PySR 2.x; the current implementation explicitly checks for a minimum version of `2.0.0a2`.

The bundled pyKAN dependency list is provided in:

```text
third_party/pykan/requirements_original.txt
```

A practical setup is to create a dedicated Python environment, install PySR, and install the numerical dependencies required by the bundled pyKAN implementation. The exact environment may be adapted to the local operating system, but the Python executable path must then be assigned to `Stage0PythonExe` in the experiment Live Scripts.

#### EQL-Div environment

EQL-Div uses a separate legacy environment. Environment specifications are included:

```text
baselines/eql/environment_official.yml
baselines/eql/environment_official_windows.yml
```

For Windows, for example:

```bash
conda env create -f baselines/eql/environment_official_windows.yml
```

The environment currently specifies Python 3.7, NumPy 1.19.5, SciPy 1.5.4, Theano 1.0.5, and Lasagne.

After creating the environment, set `EQLPythonExe` in the relevant Live Script to the Python executable of this environment.

### 1.3 Path checklist before training

Before launching a full experiment, verify:

1. MATLAB Current Folder is the repository root.
2. In the root-level `run_demo_*.mlx` file you plan to execute, `Stage0PythonExe` points to the Python environment containing PySR and the required numerical packages.
3. In the same Live Script, `EQLPythonExe` points to the dedicated EQL environment if EQL-Div will be trained.
4. `third_party/pykan/` exists.
5. `baselines/eql/official_eql/` exists.
6. The separately downloaded `outputs/` folder is present if replaying pretrained models.
7. The separately downloaded `result for paper/` folder is present if inspecting the archived paper experiment records.

---

## 2. Repository Structure

A simplified structure is:

```text
PhDN/
│
├─ baselines/
│  ├─ common/
│  ├─ sr/
│  ├─ sindy/
│  ├─ mlp/
│  ├─ kan/
│  └─ eql/
│
├─ core/
│  └─ stage0/
│
├─ system_identification/
├─ tasks/
├─ third_party/
│  └─ pykan/
├─ tools/
├─ utils/
│
├─ outputs/                         # downloaded separately
├─ result for paper/               # downloaded separately
│
├─ run_demo_feynman_dimless.mlx
├─ run_demo_single_generator_dynamic.mlx
├─ run_demo_soft_saturated_lorenz96.mlx
├─ run_demo_symbolic_representability_automatric_compiler.mlx
│
├─ replot_single_generator_dynamic_paper_figures.m
├─ replot_soft_saturated_lorenz96_paper_figures.m
└─ print_demo_output.m
```

### `baselines/`

Implementations and MATLAB/Python adapters for the comparison methods.

- `sr/`: official PySR interface and supervised process control.
- `sindy/`: SINDy and neural-augmented SINDy dictionaries and training routines.
- `mlp/`: MLP sweeps used in the comparison experiments.
- `kan/`: KAN training, pruning, refinement, and evaluation.
- `eql/`: EQL-Div adapters and the included reference implementation.
- `common/`: utilities shared by Python-based baselines.

### `core/`

Core PhDN implementation, including dictionary construction, Stage-0 SR integration, SR-to-PhDN compilation, masked least-squares initialization, Stage-1/Stage-2 refinement, pruning, prediction, and result summarization.

The most important Stage-0 logic is located under:

```text
core/stage0/
```

### `system_identification/`

Case-specific utilities for the two dynamical-system identification experiments, including:

- data generation and Sobol sampling;
- pretrained-result loading and replay;
- predictor compilation;
- trajectory rollout;
- result persistence;
- sample-efficiency summaries;
- paper-figure generation.

### `tasks/`

Definitions of the benchmark tasks and physical systems used by the main experiments.

### `third_party/`

Third-party source code bundled for reproducibility. The current package includes the pyKAN implementation used by the KAN/MLP baselines.

### `tools/`

Auxiliary utilities used when preparing structural priors or initial coefficients. These scripts are not required for ordinary pretrained-result replay.

### `utils/`

Lower-level helper functions for masks, coefficient packing/unpacking, dictionary inspection, and display.

### `tests/` and `tmp/`

These folders are not required for reviewing the paper results.

- `tests/` contains development/debugging tests.
- `tmp/` stores temporary intermediate files generated during MATLAB–Python execution.

They are intentionally omitted from the compact public data package and can be regenerated when needed.

---

## 3. Main Experiment Programs

The `.mlx` files in the repository root are the main experiment entry points.

### Feynman relationship-learning experiment

```text
run_demo_feynman_dimless.mlx
```

Runs the Feynman-style relationship-learning benchmark.

### Symbolic representability experiment

```text
run_demo_symbolic_representability_automatric_compiler.mlx
```

Demonstrates constructive symbolic representability and automatic compilation into a PhDN architecture.

### Single-generator system identification

```text
run_demo_single_generator_dynamic.mlx
```

Runs the salient-pole single-generator dynamic-system identification experiment, including vector-field identification and unseen-initial-condition trajectory rollout.

### Soft-saturated Lorenz–96 system identification

```text
run_demo_soft_saturated_lorenz96.mlx
```

Runs the soft-saturated Lorenz–96 identification experiment and its sample-efficiency/trajectory evaluations.

---

## 4. Supplementary Experiment Data

The large data package is distributed separately:

**Download:** https://drive.google.com/file/d/1uoIL0bR3r9Y7qatRmm4FYaplonLZV_OR/view?usp=sharing  https://drive.google.com/file/d/1rXuBUDSU4PxXsUlEuBx9bF2gf5TabFxR/view?usp=sharing

After downloading, copy the following folders into the repository root:

```text
outputs/
result for paper/
```

The expected layout is therefore:

```text
PhDN/
├─ outputs/
├─ result for paper/
├─ baselines/
├─ core/
├─ system_identification/
├─ tasks/
├─ ...
└─ run_demo_*.mlx
```

Do not place `outputs/` or `result for paper/` one directory above or below the repository root, because the supplied scripts construct paths relative to the current project root.

---

## 5. `outputs/`: Trained Models and Replay Data

`outputs/` contains machine-readable saved models and result structures used for fast reproduction.

The current system-identification cases are organized approximately as:

```text
outputs/
├─ SingleGeneratorDynamic_SMIB_AVR/
│  ├─ round_01/
│  │  ├─ N_00250/
│  │  ├─ N_00500/
│  │  ├─ N_01000/
│  │  └─ N_02000/
│  └─ summary/
│
└─ SoftSaturatedLorenz96_K10_F8_kappa1/
   ├─ round_01/
   │  ├─ N_00250/
   │  ├─ N_00500/
   │  ├─ N_01000/
   │  └─ N_02000/
   └─ summary/
```

Each sample-size directory contains persisted method results/checkpoints. The `summary/` folder contains aggregate results used to reconstruct comparison tables and figures.

These files are intended mainly for programmatic replay and normally do not need to be edited manually.

---

## 6. `result for paper/`: Human-Readable Experiment Records

`result for paper/` stores the detailed experiment records supporting the manuscript.

The folders are organized by experiment section, for example:

```text
result for paper/
├─ Feynman learning/
├─ Symbolic Representability/
└─ System Identification/
```

The archived MATLAB Live Scripts/reports contain substantially more information than can be included in the paper, such as:

- complete experiment settings;
- sample sizes and random seeds;
- structural-prior definitions;
- Stage-0 SR search reports;
- selected symbolic candidates;
- PhDN architecture/training diagnostics;
- baseline settings;
- numerical metrics;
- detailed trajectory simulations;
- auxiliary figures and intermediate results.

Readers who want to audit an experiment in detail should inspect these records in addition to the manuscript.

The `.mlx` files can be opened directly in MATLAB. Other software capable of reading MATLAB Live Scripts may also be used, although MATLAB is recommended because it preserves the original formatted outputs and figures.

---

## 7. Fast Reproduction Using Pretrained Models

The two system-identification demos are distributed in a **replay-oriented default mode** so that readers can reproduce the principal dynamic simulations and figures without repeating the expensive training procedure.

For:

```text
run_demo_single_generator_dynamic.mlx
run_demo_soft_saturated_lorenz96.mlx
```

the normal pretrained replay pattern is:

```matlab
SaveResults = false;

RunPhDNMainModel_G1 = false;
RunPhDNMainModel_G2 = false;
RunPhDNMainModel_G3 = false;

RunMLPBaseline = false;
RunEQLBaseline = false;
RunKANBaseline = false;
RunSINDyBaseline = false;
```

For the Lorenz–96 case, the neural-SINDy training switch should also remain disabled during replay:

```matlab
RunNeuralSINDyBaseline = false;
```

The corresponding `displayRecordReport_*` switches determine which saved methods are loaded and included in the current replay. For example:

```matlab
displayRecordReport_PhDN_G1 = true;
displayRecordReport_PhDN_G2 = true;
displayRecordReport_PhDN_G3 = true;

displayRecordReport_MLP = true;
displayRecordReport_EQL = true;
displayRecordReport_KAN = true;
displayRecordReport_SINDy = true;
```

The Lorenz–96 demo additionally supports:

```matlab
displayRecordReport_NeuralSINDy = true;
```

When these replay switches are enabled, the stored trained models are loaded from `outputs/`, the dynamical rollouts are recomputed, and the comparison summaries/figures can be reconstructed without performing the original training searches.

`RecordedReportCompactMode = true` is recommended for routine replay because it avoids printing very large historical search reports while leaving the archived results unchanged.

---

## 8. Retraining the Models

To reproduce the training process rather than replay the stored models, enable only the methods that should be retrained.

For example, to retrain the three PhDN prior levels:

```matlab
SaveResults = true;

RunPhDNMainModel_G1 = true;
RunPhDNMainModel_G2 = true;
RunPhDNMainModel_G3 = true;

displayRecordReport_PhDN_G1 = false;
displayRecordReport_PhDN_G2 = false;
displayRecordReport_PhDN_G3 = false;
```

Keep unrelated baselines disabled unless they also need to be retrained:

```matlab
RunMLPBaseline = false;
RunEQLBaseline = false;
RunKANBaseline = false;
RunSINDyBaseline = false;
```

For Lorenz–96:

```matlab
RunNeuralSINDyBaseline = false;
```

To reproduce the current full-data Stage-0 SR protocol, retain:

```matlab
Stage0Batching = false;
Stage0BatchSizeList = TrainingSampleList;
```

Before training, verify all Python executable paths carefully.

### Important runtime note

Full PhDN/SR/KAN/EQL training can be computationally expensive. The provided pretrained checkpoints are therefore the recommended starting point for readers who only want to reproduce the reported trajectories, plots, and summary metrics.

---

## 9. Replotting Paper Figures

For the system-identification experiments, paper figures can also be regenerated from saved aggregate results without rerunning training.

Convenience scripts are provided:

```text
replot_single_generator_dynamic_paper_figures.m
replot_soft_saturated_lorenz96_paper_figures.m
```

The main system-identification Live Scripts also provide a plot-only mode through:

```matlab
RegeneratePaperFiguresOnly = true;
```

When enabled, the script loads the existing aggregate summary and regenerates the paper figures without training, saved-model replay, or trajectory reevaluation.

---

## 10. Recommended Reproduction Workflow

For most readers, the following sequence is recommended:

1. Download and extract the GitHub repository.
2. Download the supplementary data package from **[DATA_DOWNLOAD_LINK]**.
3. Copy `outputs/` and `result for paper/` into the repository root.
4. Open MATLAB and set the repository root as the Current Folder.
5. Review the corresponding archived record under `result for paper/` for the exact experimental settings and detailed report.
6. Run the desired root-level `.mlx` demo in pretrained replay mode.
7. Compare the regenerated trajectories, sample-efficiency summaries, and figures with those reported in the manuscript.
8. Only if full retraining is desired, configure the Python environments and executable paths, enable the corresponding `Run*` switches, and rerun the experiment.

---

## 11. Notes on Reproducibility

- Training and validation/test data are generated according to the settings stored in each experiment driver and archived report.
- Symbolic-regression search is stochastic unless strict deterministic controls are explicitly enabled; small differences in discovered expressions may therefore occur across operating systems, Python/Julia environments, or thread scheduling.
- Saved pretrained models are provided to make the paper-level numerical results directly inspectable and reproducible without relying on a new stochastic search.
- For exact parameter settings used in a particular reported experiment, the corresponding record under `result for paper/` should be treated as the primary reference.

---


