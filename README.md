# GNN-NPN

This repository contains the official implementation of the paper:

> **GNN-NPN: Efficient NPN Classification for Large Functions using Graph Neural Network**
>
> Chengdi Cao, Hanyu Wang, Wan-Hsuan Lin, and Jason Cong (UCLA CS)
>
> *IEEE/ACM International Conference on Computer-Aided Design (ICCAD ’26)*

## Abstract

Two Boolean functions are considered NPN equivalent if one can be transformed into the other by negating inputs, permuting inputs, or negating the output. Efficient NPN classification is valuable in fields such as logic synthesis, quantum computing, and information theory. However, exact NPN classification demands exhaustive enumeration of all possible input permutations and negations. Its complexity grows exponentially, making it impractical for larger functions with more inputs. In this paper, we introduce the first learning-based formulation for NPN classification by recasting it as NPN action prediction. Our method represents Boolean functions as AND-Inverter Graphs (AIGs) and applies a bidirectional heterogeneous graph neural network to predict input negations, output negation, and input permutations. To mitigate the non-canonicity of logic networks, we further introduce graph rewriting techniques that reduce structural ambiguity and recover useful symmetries before prediction. Experimental results show that NPN action prediction is learnable, that the resulting canonicalization quality improves over heuristic baselines, and that on practical benchmarks our method approaches exact methods on tractable cases while scaling to functions with up to 16 inputs.

## Repository Structure

```
GNN-NPN/
├── main.py             # entry point: training, evaluation, inference, NPN classification
├── prepare_data.py     # data preparation: cut EPFL circuits, rewrite cuts, simulate truth tables
├── configs/            # YAML experiment configs (EPFL cuts / random AIGs)
├── scripts/            # shell scripts that run each pipeline step over 6–16-input cuts
├── src/                # model, datasets, supervised and GRPO training, NPN evaluation
├── wrapper/            # pybind11 bindings exposing mockturtle to Python (AIG / truth-table ops)
├── CMakeLists.txt      # builds the wrapper
├── examples/           # small example AIGs
├── third-party/
│   ├── benchmarks/     # EPFL combinational benchmark suite (source circuits)
│   ├── mockturtle/     # mockturtle logic synthesis library (header-only)
│   └── python-deepgate/  # DeepGate2 GNN, lightly modified, used as the AIG encoder
└── requirements.txt    # Python dependencies
```

The following directories are ignored by git but are created and filled when the
code runs:

| Directory | Written by | Contents |
| --- | --- | --- |
| `build/` | CMake | Build tree of the mockturtle wrapper. The compiled module is `build/python/mockturtle_wrapper*.so`, which `src/mockturtle.py` finds automatically. |
| `benchmarks/` | `prepare_data.py` | Prepared data, laid out as `benchmarks/<circuit>/<num_inputs>/`: `aig/` holds the cuts extracted from each EPFL circuit, `aig_rewriting/` the same cuts after graph rewriting, and `truth/` their simulated truth tables. Not to be confused with `third-party/benchmarks/`, the original circuits. |
| `runs/` | `main.py train-supervised`, `train-grpo` | One directory per training run, e.g. `runs/epfl/6bit/supervised/` or `runs/random/6bit/grpo/`. Each holds the resolved `config.yaml`, the `best.pt` and `last.pt` checkpoints, and TensorBoard event files. |
| `results/` | `main.py uniquify-truth-tables`, `run-inference` | NPN classification inputs and outputs, per input count `<b>`: `epfl_truth_tables_<b>bit.txt` (the unique truth tables), `epfl_unique_indices_<b>bit.txt` (the dataset index of one cut per unique truth table), and `epfl_inference_results_<b>bit.pt` (the saved policy logits that `run-npn` reads). |

Output paths are set in the config files and can be overridden on the command
line (e.g. `supervised.dir=...`, `npn.inference_results_path=...`).

## Environment Setup

Requirements:

- Python ≥ 3.10
- A C++17 compiler and CMake ≥ 3.18, to build the mockturtle wrapper
- A GPU is optional; it is used automatically when PyTorch detects one

We tested with Python 3.10, PyTorch 2.9.1, PyTorch Geometric 2.8.0, GCC 12.1
and CMake 3.26.

### 1. Install Python dependencies

All Python dependencies are listed in [`requirements.txt`](requirements.txt).
We recommend installing them into a fresh virtual environment:

```sh
pip install -r requirements.txt
```

If you need a PyTorch build for a specific CUDA version, install it first by
following the [PyTorch installation guide](https://pytorch.org/get-started/locally/),
then run the command above.

### 2. Build the mockturtle wrapper

Run this from the repository root with the same Python environment active, since
the build uses that environment's Python and `pybind11`:

```sh
cmake -S . -B build \
    -DPython3_EXECUTABLE="$(which python)" \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build -j
```

Neither the wrapper nor DeepGate2 needs to be installed with `pip`: `src/` loads
them from `build/python/` and `third-party/python-deepgate/` respectively. To
check that both load:

```sh
python -c "import src.mockturtle, src.deepgate; print('ok')"
```

## Data Preparation

Training and evaluation use small cuts extracted from the circuits of the EPFL
benchmark suite in `third-party/benchmarks/`. The scripts in `scripts/` prepare
them for every input count from 6 to 16 in steps of 2. Run them from the
repository root in this order:

```sh
# 1. Extract cuts from every EPFL circuit
bash scripts/cut_aig.sh

# 2. Both need only step 1, and can run in either order
bash scripts/rewrite_aig.sh
bash scripts/simulate_tt.sh

# 3. Needs the truth tables from simulate_tt
bash scripts/uniquify_truth_tables.sh
```

Each step writes into `benchmarks/<circuit>/<k>/`, where `<k>` is the number of
inputs of the cuts:

- **`cut_aig.sh`** extracts reconvergence-driven cuts with `k` leaves from each
  circuit and saves them as AIGs in `aig/`. This gives about 2.2 million cuts in
  total, between 330k and 400k for each `k`.
- **`rewrite_aig.sh`** applies graph rewriting to each cut and saves the result
  in `aig_rewriting/`. These are used when a dataset config sets
  `rewriting: true`.
- **`simulate_tt.sh`** simulates the truth table of each cut into `truth/`.
  Rewriting does not change a cut's function, so these truth tables serve the
  rewritten cuts as well. They are optional for training, which simulates any
  missing truth table on the fly, but they make data loading faster.
- **`uniquify_truth_tables.sh`** collects the distinct truth tables among the
  cuts of each `k`. These are the functions to be NPN-classified. It writes
  `results/epfl_truth_tables_<k>bit.txt` and, for each truth table, the dataset
  index of one cut that implements it to `results/epfl_unique_indices_<k>bit.txt`.
  Extra `key=value` arguments are passed on to `main.py` as config overrides.

Together, the steps write about 6.5 million small files to `benchmarks/`, taking
about 32 GB of disk space.

To prepare a single input count, call `prepare_data.py` directly, e.g.
`python prepare_data.py cut_aig --cut-size 8`, and likewise for `rewrite_aig`
and `simulate_tt`. Its `--data-dir` and `--output-dir` options change where it
reads circuits from and writes to.

The random-AIG dataset (`configs/random_6bit.yaml`) is generated on the fly
during training and needs no preparation.

## Running `main.py`

Training, evaluation and NPN classification all go through `main.py`:

```sh
python main.py <command> <config.yaml> [key=value ...]
```

### Commands

| Command | What it does |
| --- | --- |
| `train-supervised` | Trains the model to predict the NPN action that the algorithm in `dataset.<i>.action` computes for each function. Saves to `supervised.dir`. |
| `test-supervised` | Reports loss and accuracy against those labels on the test split of each dataset. |
| `train-grpo` | Trains the model with GRPO. The reward of an action is how much it lowers the truth table's value, since the exact NPN representative is the smallest truth table in its class. Saves to `reinforcement.grpo.dir`. |
| `test-reinforcement` | Samples `reinforcement.group_size` actions per function on the test split of each dataset and reports their mean and best reward. |
| `uniquify-truth-tables` | Collects the distinct truth tables of a dataset (see [Data Preparation](#data-preparation)). |
| `run-inference` | Runs the model on one cut per distinct truth table (listed in `npn.indices_path`) and saves its output logits to `npn.inference_results_path`. |
| `run-npn` | Groups the distinct truth tables in `npn.truth_table_path` into NPN classes using the saved logits, and prints the number of classes found. Needs no dataset or model. |

Commands that use the model start from random weights unless `load` points to a
checkpoint, so the `test-*` commands and `run-inference` are normally run with
`load=<run dir>/best.pt`.

`run-npn` samples `npn.trial` actions per truth table and merges truth tables
whose actions reach a common result. Every merge is between NPN-equivalent
functions, so the true number of NPN classes is a lower bound on the count
reported, and a lower count is better.

### Configuration

`configs/` has two example configs: `epfl_6bit.yaml` uses the 6-input cuts of
the EPFL circuits and `random_6bit.yaml` uses random 6-input AIGs. Training
commands save the full config, including overrides, to `config.yaml` in their
output directory.

| Key | Meaning |
| --- | --- |
| `seed` | Random seed. It also fixes the EPFL train/val/test split and the random AIGs. |
| `num_workers` | Threads used to apply actions to truth tables. Defaults to all available CPU cores when unset. |
| `load` | Checkpoint to initialize the model from; `null` for random weights. |
| `dataset` | List of datasets. Training uses all of them together; validation and testing report each one separately. |
| `dataset.<i>.cls` | `epfl` for the prepared cuts, `random` for random AIGs. |
| `dataset.<i>.name` | Name shown in logs and TensorBoard. |
| `dataset.<i>.num_bit` | Number of inputs of each function. |
| `dataset.<i>.rewriting` | Use the rewritten AIGs instead of the original ones. |
| `dataset.<i>.action` | Algorithm that computes the supervised labels: `exact_npn`, `exact_n`, `exact_p`, `flip_swap_npn` or `sifting_npn`. Needed by the supervised commands; leave it `null` otherwise, since computing labels takes time. |
| `dataset.<i>.root`, `train_split`, `val_split` | EPFL only: the prepared data directory, and the fractions of cuts used for training and validation. The remaining cuts form the test split. |
| `dataset.<i>.num_train`, `num_eval`, `num_test` | Random only: AIGs generated per training epoch, and the sizes of the fixed validation and test sets. |
| `dataset.<i>.mean_nodes`, `std_nodes` | Random only: mean and standard deviation of the number of AND gates per AIG. |
| `agent.num_rounds` | Rounds of message passing in the GNN. |
| `agent.dim_hidden` | Hidden dimension of the GNN. |
| `agent.enable_reverse` | In each round, also propagate messages from the output back to the inputs. |
| `agent.enable_layernorm` | Apply LayerNorm to the node states after each round. |
| `supervised.*` | Supervised training: `train_steps`, `batch_size`, `lr`, `grad_clip`, `log_interval` and `eval_interval` (in steps), and the output `dir`. |
| `reinforcement.batch_size`, `group_size` | Functions per batch, and actions sampled per function. |
| `reinforcement.temp` | Sampling temperature; `0` always takes the most likely action. |
| `reinforcement.post_proc` | Heuristic canonization applied to the truth table after the model's action (same choices as `action`, or `null`). Rewards are computed both with it (`r_*`) and without it (`r_raw_*`). |
| `reinforcement.grpo.raw_reward` | Train on the reward without post-processing (`true`) or with it (`false`). This also decides which validation reward selects `best.pt`. |
| `reinforcement.grpo.num_epoch` | Policy updates per batch of sampled actions. |
| `reinforcement.grpo.clip_eps`, `ent_coef` | PPO clipping range and entropy bonus weight. |
| `reinforcement.grpo.*` (others) | `train_steps`, `lr`, `grad_clip`, `log_interval`, `eval_interval` and `dir`, as for supervised training. |
| `npn.trial`, `temp`, `post_proc` | `run-npn`: actions sampled per truth table, sampling temperature, and post-processing. |
| `npn.truth_table_path`, `indices_path` | The distinct truth tables and their cut indices, written by `uniquify-truth-tables`. |
| `npn.inference_batch_size`, `inference_results_path` | `run-inference`: batch size and output file. |

### Overriding config values

Any config value can be overridden with `key=value` arguments after the config
file:

- Keys are dotted paths, and list entries are selected by position, e.g.
  `dataset.0.num_bit`.
- Values are parsed the same way as in the config file: `8` is an integer,
  `1e-5` a float, `true` a boolean, `null` empty, and anything else a string.
- A key that is not in the config prints a warning, since it is most likely a
  typo, and is then added to the config.

For example, to train on 8-input EPFL cuts with labels from `sifting_npn` and
then evaluate the best checkpoint:

```sh
python main.py train-supervised configs/epfl_6bit.yaml \
    dataset.0.name=epfl_8bit dataset.0.num_bit=8 dataset.0.action=sifting_npn \
    supervised.dir=runs/epfl/8bit/supervised

python main.py test-supervised runs/epfl/8bit/supervised/config.yaml \
    load=runs/epfl/8bit/supervised/best.pt
```

Evaluating with the saved `config.yaml` reuses the training run's seed and split
settings, so the test split is the same one that was held out during training.

> **Note:** Step-by-step instructions for training and evaluation will be added soon.

## License

See [LICENSE](LICENSE).
