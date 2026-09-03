# QSVM Encoding Benchmark

A benchmarking framework for comparing quantum data-encoding strategies ("feature maps") in a
Quantum Support Vector Machine (QSVM) setting, against a classical RBF-kernel SVM baseline.

Each encoding is used to build a **precomputed kernel** (Gram matrix) via a quantum circuit
simulated with [PennyLane](https://pennylane.ai/); the kernel is then plugged into a standard
`sklearn.svm.SVC(kernel="precomputed")`. The framework handles dataset preparation,
hyperparameter search (cross-validated) per encoding, final model training, evaluation, and
result logging, for an arbitrary list of datasets and encodings.

> This README only explains **what the code does and how to use/extend it**. For experimental
> results, the comparison between encodings, and the discussion/conclusions, see the
> [`report/`](report/) directory.

## Contents

- [Repository structure](#repository-structure)
- [Requirements & setup](#requirements--setup)
- [Quick start](#quick-start)
- [Pipeline: what `qsvm()` does](#pipeline-what-qsvm-does)
- [Configuration reference](#configuration-reference)
- [Adding / removing a dataset](#adding--removing-a-dataset)
- [Adding / removing a kernel (encoding)](#adding--removing-a-kernel-encoding)
- [Simulation notes](#simulation-notes)
- [Encodings implemented](#encodings-implemented)
- [Output](#output)

## Repository structure

```
encoding_benchmark.ipynb     Entry point notebook: defines the config, the datasets and
                              the list of kernels, and runs the benchmark.

modules/
  main.py                    qsvm(): orchestrates a full benchmark run for one dataset
                              against a list of kernels.
  dataset_preparation.py     Undersampling, train/test split, PCA.
  training.py                Hyperparameter grid search + cross-validation, final refit
                              of the best model.
  diagnostics.py             Test-set evaluation report, confusion matrix and Gram matrix
                              plots.
  metrics.py                 Kernel-Target Alignment (KTA) and geometric coefficient.
  quantum_devices.py         Factory functions that build the PennyLane device used by
                              every kernel.
  kernels/
    kernel_rbf.py             Classical RBF baseline (always included automatically).
    kernel_amplitude.py       Amplitude encoding.
    kernel_angle.py            Angle encoding.
    kernel_basis.py            Basis encoding.
    kernel_IQP.py               IQP encoding.
    kernel_projected.py        Projected quantum kernel.

report/                      The written report (results, comparison, discussion).
run_results/                 Auto-generated output of each run (git-ignored).
```

## Requirements & setup

The project targets Python 3.12. The exact dependency list used in the provided dev
container is in [`.devcontainer/Dockerfile`](.devcontainer/Dockerfile); the core packages are:

```
numpy scipy pandas matplotlib seaborn
scikit-learn
pennylane pennylane-lightning
tabulate tqdm openpyxl jupyterlab notebook
```

Easiest path: open the repository in the provided **dev container**
(`.devcontainer/`), which builds a ready-to-use image with everything installed and starts
JupyterLab. Alternatively, install the packages above with `pip` in your own environment and
open `encoding_benchmark.ipynb`.

## Quick start

```python
from modules.quantum_devices import lightning_qubit_device
from modules.kernels import kernel_amplitude, kernel_angle, kernel_basis, kernel_IQP, kernel_projected
from modules.main import qsvm

config = {
    "number_samples": 200,      # samples drawn from the dataset (undersampled if larger)
    "number_features": 4,       # target dimensionality after PCA (upper bound, see below)
    "fraction_train": 0.80,     # train/test split fraction
    "max_qubits": 4,            # qubit budget available to the kernels
    "quantum_device": lightning_qubit_device,
}

kernel_modules = [kernel_amplitude, kernel_angle, kernel_basis, kernel_IQP, kernel_projected]

dataset = {
    "dataset_name": "my_dataset",
    "dataset_original_data": (X, y),   # X: ndarray (n_samples, n_features), y: ndarray of {0, 1}
}

qsvm(dataset, config, kernel_modules)
```

This is exactly the pattern used in `encoding_benchmark.ipynb`, repeated once per dataset. Each
call to `qsvm(...)` is fully independent — it takes one dataset, one config, and a list of
kernels, and produces one set of results.

## Pipeline: what `qsvm()` does

For a single call `qsvm(dataset, config, kernel_modules)`:

1. **Baseline injection** — the classical RBF kernel (`kernel_rbf`) is always prepended to
   `kernel_modules`, regardless of what you pass in, so every run has a classical reference
   point.
2. **Dataset preparation** (`dataset_preparation.prepare_dataset`) — undersamples to
   `number_samples` (stratified), splits into train/test (`fraction_train`, stratified), and
   caps `number_features` to the dataset's actual number of raw features if it requests more
   than are available (a warning is printed if this happens).
3. **Per kernel, in order** (classical RBF first):
   - `get_kernel_hyperparameters(number_features, max_qubits)` builds the hyperparameter grid
     specific to that encoding (e.g. `gamma` for RBF, `number_repeats`/`pattern` for IQP).
   - **Cross-validated grid search** (`training.train_model`): 5-fold stratified CV (reduced
     automatically if a class has fewer than 5 samples) over the kernel's own hyperparameters
     combined with the SVM's `C`. PCA is fit independently per fold to avoid data leakage.
     Scoring is macro-F1 on the validation folds.
   - The best combination is **refit on the full training set** to produce the final kernel
     module's `SVC`.
   - **Evaluation** (`diagnostics.evaluate_model`) on the held-out test set: classification
     report, confusion matrix, and a Gram-matrix heatmap (sorted by class).
   - **Metrics** (`metrics.py`): Kernel-Target Alignment (KTA) for every kernel, and — for every
     non-classical kernel — a geometric coefficient against the classical RBF kernel's Gram
     matrix, as a proxy for how geometrically different the quantum kernel is from the
     classical one.
4. **Logging** — an Excel summary is written and updated after every kernel, plus a final
   summary once all kernels are done (see [Output](#output)).

## Configuration reference

| Key | Meaning |
|---|---|
| `number_samples` | Number of samples drawn (stratified) from the dataset before splitting. Must not exceed the dataset's size. |
| `number_features` | Target number of features after PCA. Treated as an **upper bound**: if the dataset has fewer raw features, all of them are used instead (with a printed note), no error is raised. |
| `fraction_train` | Fraction of `number_samples` used for training (the rest is the test set). |
| `max_qubits` | Qubit budget passed to every kernel's `get_kernel_hyperparameters`. Only `kernel_basis` currently validates it directly, because it's the only encoding whose qubit count (`number_features * tau_bit_feature`) isn't already implied by `number_features` alone — for the others, keeping `number_features <= max_qubits` in the config is enough to stay within budget. |
| `quantum_device` | A factory function `number_qubits -> qml.Device`, from `modules/quantum_devices.py` (`lightning_qubit_device`, or `allocate_GPU_device` to offload to GPU above a qubit threshold). You can add your own following the same signature. |

## Adding / removing a dataset

A dataset is just a dict:

```python
dataset = {
    "dataset_name": "some_name",                # used in filenames/folders/plot titles
    "dataset_original_data": (X, y),             # X: ndarray (n_samples, n_features)
                                                  # y: ndarray of exactly two classes, labeled 0/1
}
```

**Binary classification only** — `y` must have exactly two classes labeled `0`/`1`; KTA
calculation and the per-class reporting in `main.py` both assume this.

To use it, call `qsvm(dataset, config, kernel_modules)`. To remove a dataset from a benchmark
run, simply don't call `qsvm()` for it — there's no registry to edit.

## Adding / removing a kernel (encoding)

A kernel is a Python module exposing three things:

```python
kernel_name = "my_kernel"   # used in filenames/folders/plot titles/reports

def get_kernel_hyperparameters(number_features, max_qubits):
    # Return a dict of lists (fed to sklearn's ParameterGrid) describing the
    # hyperparameter grid to search for this encoding. Return {} if there is none.
    # If your qubit count depends on more than `number_features` alone (like kernel_basis's
    # tau_bit_feature), validate it against max_qubits here.
    return {"my_hyperparam": [0.1, 1.0, 10.0]}

def calculate_kernel(X_dataset_1, X_dataset_2, number_features, quantum_device, my_hyperparam):
    # Build the PennyLane circuit, embed X_dataset_1 / X_dataset_2, and return the Gram
    # matrix of shape (len(X_dataset_1), len(X_dataset_2)).
    # `quantum_device` is the factory from the config: quantum_device(number_qubits).
    ...
    return kernel
```

Look at `modules/kernels/kernel_angle.py` for the simplest complete example, and
`modules/kernels/kernel_basis.py` or `kernel_projected.py` for encodings with tunable
hyperparameters. A couple of conventions the existing kernels follow (not strictly required,
but recommended for consistency and performance):

- Fit any scaler (`MinMaxScaler`, etc.) on `X_dataset_1` only, and clip `X_dataset_2` after
  transforming it, to avoid leaking test-set information into the encoding's preprocessing.
- Short-circuit the case `X_dataset_1 is X_dataset_2` (or `np.array_equal`) with
  `qml.kernels.square_kernel_matrix(..., assume_normalized_kernel=True)` instead of the full
  `qml.kernels.kernel_matrix`, since this is the common case (computing `K_train`) and it
  roughly halves the number of circuit evaluations.

To use it, import the module and add it to the `kernel_modules` list passed to `qsvm()`. To
remove an encoding, just leave it out of that list — the classical RBF baseline is the only
one that can't be removed, since it's injected automatically and is the reference point for the
geometric coefficient.

## Simulation notes

All kernels are evaluated with **exact, noiseless statevector simulation**
(`lightning.qubit` / `lightning.gpu`, no `shots=` set anywhere): every kernel value is an
analytic expectation value, with no sampling noise and no hardware noise model. This keeps the
quantum kernels' Gram matrices provably PSD (as required for `SVC(kernel="precomputed")`) without
needing extra regularization, but it does not reflect the behavior of a noisy real device —
keep that in mind when interpreting results or porting this to hardware.

## Encodings implemented

| Encoding | Qubits used | Pros | Cons |
|---|---|---|---|
| **Amplitude** (`kernel_amplitude`) | `ceil(log2(number_features))` | Extremely qubit-efficient — an entire feature vector fits into one state. | Needs a state-preparation circuit whose depth grows with the number of amplitudes (not NISQ-friendly beyond simulation); encodes only the *direction* of the feature vector, the norm is normalized away and lost. |
| **Angle** (`kernel_angle`) | `number_features` | Shallow, simple circuit (one rotation per feature), no entanglement, easy to reason about. | Qubit count scales linearly with the number of features, so it doesn't scale to high-dimensional data; with no entanglement, each feature is encoded independently, which limits kernel expressivity. |
| **Basis** (`kernel_basis`) | `number_features * tau_bit_feature` | Conceptually the simplest encoding — a deterministic, exact bit-exact mapping. | It's an **exact-match kernel**: two samples are "similar" (kernel = 1) only if their quantized bitstrings coincide exactly, otherwise it's 0 — there is no real notion of distance/smoothness. At `tau_bit_feature = 1` (often the only value that fits the qubit budget) each feature is reduced to a single bit, which is a very reductive quantization. Qubit cost also grows the fastest of all encodings here. |
| **IQP** (`kernel_IQP`) | `number_features` | Based on circuit families believed to be classically hard to simulate; tunable expressivity via repeats and entangling pattern. | More hyperparameters to search (repeats × pattern); deep/highly-entangled instances of this family are known to suffer from *exponential concentration* (kernel values collapsing towards a constant), which can make training harder as circuits grow. |
| **Projected quantum kernel** (`kernel_projected`) | `number_features` | Designed to avoid the exponential-concentration issue of pure fidelity kernels, by projecting the state down to local expectation values (⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit) and computing a classical RBF kernel on those — generally more trainable in practice. | Projecting to few-body observables discards information, so it can lose some of the expressivity a full fidelity kernel would have; adds its own classical hyperparameters (`gamma`, number of entangling layers) to tune. |
| **Classical RBF** (`kernel_rbf`) | — (no qubits, classical) | Standard, well-understood, fast; used purely as the classical reference/baseline for every comparison. | Not the object of study — just the yardstick every quantum kernel is measured against. |

## Output

Every `qsvm(...)` call creates a folder tree under `run_results/` (git-ignored):

```
run_results/<YYYYMMDD>/<YYYYMMDD>_<dataset>/<YYYYMMDD>_<dataset>_<HHMMSS>/
  partial_results_<dataset>.xlsx        Updated after every kernel finishes.
  final_summary_<dataset>.xlsx          Written at the end; best row (by F1) highlighted.
  <..>_<kernel_name>/
    training_history_<dataset>.xlsx     Every hyperparameter combination tried during CV.
    best_model_plots_<dataset>.svg      Confusion matrix + sorted Gram matrix heatmap.
```

For the interpretation of these results — which encodings performed best, on which datasets,
and why — refer to [`report/`](report/).
