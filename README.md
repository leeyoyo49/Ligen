# LIGEN: Tabular/Conditional GANs with MLP baselines

Check `tutorial.ipynb` for tutorial

## Quick start

Install required package

```
pip install -r requirements.txt
```

Use the dispatcher (defaults to `experiment/spectral.yaml`):

```
# PointGAN pipeline
python main.py --config experiment/spectral.yaml --runner pointgan  
# FreeGAN pipeline  
python main.py --config experiment/spectral.yaml --runner freegan  
# run both
python main.py --config experiment/spectral.yaml --runner both   
```

Or call runners directly:
```
python -m runners.pointgan_runner --config experiment/spectral.yaml
python -m runners.freegan_runner --config experiment/spectral.yaml
```

## Outputs
Before you run you'll be modifying /Data, /experiment only /Data puts your data.csv, /experiment puts your config.yaml


Per‑run directories (timestamped and algorithm‑prefixed):

```
experiment/<ALG>-<name>-<YYYYmmdd_HHMMSS>/
	├── your_yaml.yaml   # Original training yaml
	├── model/           # GAN and MLP checkpoints
	├── generated_data/  # synthesized CSVs
	└── logs/            # color‑free log files; latest.log pointer

Results/
	└── example_output.csv
```

CSVs

- PointGAN: `pointgan_generated_data_p{p}_s{seed}.csv` with columns:
	[positionX, positionY, <x_cols...>]
- FreeGAN (features only): `freegan_generated_features_p{p}_s{seed}.csv` with columns:
	[<x_cols...>]

Summary metrics are appended to:

```
Results/
	pointgan-<kind>-pointgan-<name>-<ts>-results.csv
	freegan-<kind>-freegan-<name>-<ts>-results.csv
```

## Package Architecture

Ligen follows a modular architecture with clear separation of concerns:
```
main.py → runners → trainers → {models, utils, dataloader}
```

### Folder Logic and Data Flow

- **`main.py`**: Entry point that dispatches to appropriate runners based on configuration
- **`runners/`**: High-level orchestrators that manage complete training pipelines
  - `pointgan_runner.py`: Manages PointGAN training → synthesis → MLP evaluation workflow
  - `freegan_runner.py`: Manages FreeGAN training → synthesis → MLP evaluation workflow
- **`trainers/`**: Contains training logic and utilities
  - Implements the actual training loops for GANs and MLPs
  - Handles model optimization, loss computation, and checkpoint management
  - Calls into `models/`, `utils/`, and `dataloader/` as needed
- **`models/`**: Neural network model definitions
  - `pointgan.py`: PointGAN (conditional GAN) implementation
  - `freegan.py`: FreeGAN (unconditional GAN) implementation  
  - `mlp.py`: Multi-layer perceptron baseline
- **`dataloader/`**: Dataset loading and preprocessing utilities
  - `wifi.py`: Wi-Fi dataset specific processing
  - `spectral.py`: Spectral dataset specific processing
- **`utils/`**: Common utilities and helpers
  - Configuration management, logging, timing, and miscellaneous utilities
- **`experiment/`**: YAML configuration files for different experiments

## Project Layout

```
├── main.py                     # Main entry point and dispatcher
├── requirements.txt            # Python dependencies
├── LICENSE                     # Project license
│
├── Data/                       # Input datasets
│   ├── datasample_spectral.csv # Spectral dataset sample
│   └── datasample_wifi.csv     # Wi-Fi dataset sample
│
├── runners/                    # High-level pipeline orchestrators
│   ├── __init__.py
│   ├── pointgan_runner.py      # PointGAN end-to-end pipeline (train → synth → MLP)
│   └── freegan_runner.py       # FreeGAN end-to-end pipeline (train → synth → MLP)
│
├── dataloader/                 # Dataset loading and preprocessing
│   ├── __init__.py             # Dataset API selector by dataset.kind
│   ├── wifi.py                 # Wi-Fi dataset utilities and transformations
│   └── spectral.py             # Spectral dataset utilities and transformations
│
├── models/                     # Neural network model definitions
│   ├── __init__.py
│   ├── pointgan.py             # PointGAN (conditional GAN) implementation
│   ├── freegan.py              # FreeGAN (unconditional GAN) implementation
│   └── mlp.py                  # Multi-layer perceptron baseline
│
├── trainers/                   # Training loops and optimization logic
│   ├── __init__.py
│   ├── pointgan_trainer.py     # PointGAN training procedures
│   ├── freegan_trainer.py      # FreeGAN training procedures
│   ├── mlp_trainer.py          # MLP training procedures
│   └── common.py               # Shared training utilities
│
├── utils/                      # Common utilities and helpers
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── logger.py               # Logging utilities
│   ├── timing.py               # Performance timing tools
│   └── misc.py                 # Miscellaneous helper functions
│
├── experiment/                 # Configuration templates and examples
│   ├── spectral.yaml           # Spectral dataset experiment config
│   └── wifi.yaml               # Wi-Fi dataset experiment config
│
└── Results/                    # Auto-generated output directory
    └── *.csv                   # Summary metrics and evaluation results
```


## Config (YAML)

Minimal required keys (see `experiment/spectral.yaml` and `experiment/wifi.yaml`):

### Supported Models

- **PointGAN**: Conditional GAN for coordinate-based data generation
- **FreeGAN**: Unconditional GAN for tabular data generation  
- **MLP**: Multi-layer perceptron baseline with Optuna hyperparameter optimization

## Tutorial

For a comprehensive tutorial on using Ligen, see `tutorial.ipynb`. The notebook covers:

- Environment setup and installation
- Data exploration and understanding
- Configuration file structure
- Model training with sample data
- Results interpretation
- Single data inference examples
- Advanced usage tips and troubleshooting

The tutorial includes hands-on examples with both spectral sensor data and WiFi positioning datasets.


## Logging & W&B
Login in to wandb (you need to turn on in config.yaml)

```
wandb login # you'll need an wandb account with API key
```

- Console: colorized logs
- Files: `experiment/<...>/logs/` (with `latest.log`)
- Weights & Biases: set `wandb.enabled: true` in YAML (project from config)
