## Repository for *SP*

This repository contains the software and data for the paper: *Structured Pixels: Using Satellite Imageries as Interventions for Causal Effect Estimation,* accepted at Hawaii International Conference on System Sciences (HICSS) 2026

Authors: Chien Lu, and Thomas Chadefaux

---

## Environment Setup
To set up the required environment, run:
```bash
conda create --name sp_env python=3.8
conda activate sp_env
```

---

## Data Preparation
To prepare the data for experiments, follow the instructions below.

### Experiment 1 (EuroSAT)
```bash
mkdir data
cd data
wget -O EuroSAT_MS.zip "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1"
wget -O EuroSAT_RGB.zip "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"

unzip EuroSAT_MS.zip
unzip EuroSAT_RGB.zip
```
The Jupyter notebook `data_synthesize_EuroSAT.ipynb` contains the necessary steps for processing and generating the experiment data. This notebook will create the folder `EuroSAT_MS_processed`, which contains processed images, and a pickle file `simulated_meta_EuroSAT.pkl` for the experiment.

### Experiment 2 (LICS)
```bash
cd data
wget -O training.zip "https://zenodo.org/records/13742222/files/training.zip?download=1"

unzip training.zip
mv training LICS
```
Similarly, the Jupyter notebook `data_synthesize_LICS.ipynb` provides the required steps for data processing and experiment data generation. This notebook will generate the folder `LICS_processed`, which contains processed images, and a pickle file `dark_vessel_simulated_meta.pkl` for the experiment.

---

## Conducting Experiments
After data preparation, navigate back to the repository folder and execute the following commands for training different models.

### SP (Our Model)
#### Experiment 1 (EuroSAT)
Train the mean outcome model:
```bash
mkdir train_mom_eurosat
python experiments/eurosat_mom.py \
    --seed 8787 \
    --batch_size 64 \
    --max_epochs 50 \
    --learning_rate 0.0001 \
    --mom_hidden_dims 64 32 \
    --mom_dropout_rate 0.5
```

Train representation learning models:
```bash
python experiments/eurosat_emb.py \
    --seed 8787 \
    --batch_size 64 \
    --max_epoch 50 \
    --learning_rate 0.0001 \
    --feature_dim 16 \
    --mom_hidden_dims 64 32 \
    --mom_dropout_rate 0.2 \
    --cov_hidden_dims 64 32 \
    --cov_dropout_rate 0.2 \
    --treatment_in_channels 9 \
    --treatment_dropout_rate 0.2 \
    --treatment_feature_model 16
```

#### Experiment 2 (LICS)
```bash
mkdir train_mom_lics
python experiments/lics_mom.py \
    --seed 8787 \
    --batch_size 16 \
    --max_epochs 50 \
    --learning_rate 0.0001 \
    --mom_hidden_dims 64 32 \
    --mom_dropout_rate 0.2

python experiments/lics_emb.py \
    --seed 8787 \
    --batch_size 16 \
    --max_epoch 50 \
    --learning_rate 0.0001 \
    --feature_dim 16 \
    --mom_hidden_dims 64 32 \
    --mom_dropout_rate 0.2 \
    --cov_hidden_dims 64 32 \
    --cov_dropout_rate 0.2 \
    --treatment_in_channels 7 \
    --treatment_dropout_rate 0.2 \
    --treatment_feature_model 16
```

---

## Baseline Models
### CNN
#### Experiment 1 (EuroSAT)
```bash
python experiments/eurosat_concat.py \
    --seed 8787 \
    --batch_size 64 \
    --max_epochs 50 \
    --learning_rate 0.0001 \
    --cov_feature_dim 16 \
    --cov_hidden_dims 64 32 \
    --cov_dropout_rate 0.2 \
    --treatment_in_channels 9 \
    --treatment_feature_dim 16 \
    --treatment_dropout_rate 0.2 \
    --outcome_hidden_dims 16 \
    --outcome_dropout_rate 0.2
```

#### Experiment 2 (LICS)
```bash
python experiments/lics_concat.py \
    --seed 8787 \
    --batch_size 16 \
    --max_epochs 50 \
    --learning_rate 0.0001 \
    --cov_feature_dim 16 \
    --cov_hidden_dims 64 32 \
    --cov_dropout_rate 0.2 \
    --treatment_in_channels 7 \
    --treatment_feature_dim 16 \
    --treatment_dropout_rate 0.2 \
    --outcome_hidden_dims 16 \
    --outcome_dropout_rate 0.2
```

### GraphITE, NICE-VGG, and NICE-ResNet
Experiments for these models follow the same format as above. Replace `concat` with `mod_graphite`, `nice_vgg`, or `nice_resnet` in the script names as needed.

---

## Notes
- Ensure that all dependencies are installed before running experiments.
- Modify hyperparameters as needed for different configurations.
- Results are stored in the respective experiment folders.