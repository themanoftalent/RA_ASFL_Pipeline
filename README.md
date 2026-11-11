# Reliability-Aware Adaptive Split Federated Learning with Uncertainty-Guided Personalization for Non-Independent and Non-Identically Distributed Electrocardiogram Classification in Internet of Medical Things Networks

This project examines how medical devices and healthcare institutions can train an electrocardiogram classifier together while keeping their recordings at their own locations. The training procedure adjusts the division of work between each client and the server. It also considers signal quality, prediction uncertainty, and the stability of local updates when combining models.

The code runs experiments in Python using simulated clients and network conditions. It is intended for research, not clinical use or deployment on medical devices.

## Data

The datasets are available from PhysioNet:

- [Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [Twelve-lead Electrocardiogram Database](https://physionet.org/content/ptb-xl/1.0.3/)

No recordings are included in this repository. Download the datasets from their original sources and place them in `data/mitdb/` and `data/ptb-xl/1.0.3/`, respectively. These folders are excluded from version control.

The first dataset is divided by recording groups. Records 201 and 202 are kept together, but the available identifiers do not establish that every other recording belongs to a different patient. The second dataset uses its official patient-preserving training, validation, and test folds.

To check the dataset links without downloading recordings, run:

```bash
python -m scripts.data_availability
```

## Installation

Use Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Experiments

Run training on either dataset:

```bash
python run.py --config configs/mitbih.yaml --out results/mitbih/ra_asfl
python run.py --config configs/ptbxl.yaml --out results/ptbxl/ra_asfl
```

Run the available learning methods with matched random seeds:

```bash
python -m scripts.experiment_matrix --config configs/mitbih.yaml --out results/comparison/mitbih
python -m scripts.experiment_matrix --config configs/ptbxl.yaml --out results/comparison/ptbxl
```

Review the completed runs:

```bash
python -m scripts.validate_runs results/comparison/mitbih --require-matched
python -m scripts.summarize results/comparison/mitbih --metric macro_f1
```

Each run saves its settings, evaluation metrics, training history, and model checkpoint in the chosen results folder. Existing experiment files are protected against accidental replacement. A separate software check uses randomly generated signals; its output is not a medical experiment.

## Code

The main modules are organized by task. `data.py` loads recordings and assigns clients. `model.py` defines the neural network. `quality.py` measures signal quality, while `controller.py` selects where to divide the network. `reliability.py` calculates uncertainty, distribution differences, and aggregation weights. `train.py` handles training and local adaptation. The remaining modules provide evaluation, privacy analysis, and experiment summaries.

## Current experimental coverage

The training program includes federated averaging, fixed split learning, resource-based adaptive split learning, and the proposed reliability-aware method. Device processing capacity, network transfer time, and energy use are simulated rather than measured on physical equipment.

The current training loop uses differences in label distributions to assess client heterogeneity. The code also contains a function for comparing feature distributions, but that function is not yet connected to the training loss. Learning the reliability coefficients, running the full privacy attack protocol, and training every comparison method discussed in the article require further implementation.

Experimental conclusions should be based on completed runs and their saved outputs. No accuracy values or other experimental results are built into the training code.

## License

The source code is released under the [software license](LICENSE.md). The electrocardiogram datasets are not part of this license. Their access and reuse conditions are provided on the PhysioNet pages linked above.
