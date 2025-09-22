import pickle, argparse

import os, sys
sys.path.append(os.getcwd())

import pandas as pd

from omegaconf import OmegaConf
from src.img_ite import *
from src.data import *
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

# Create the parser
parser = argparse.ArgumentParser(description="args for training modified graphite baseline")

# Add arguments
parser.add_argument("--seed", type=int, help="seed for experiments", default=64)
parser.add_argument("--batch_size", type=int, help="batch size", default=32)
parser.add_argument("--max_epochs", type=int, help="training_epochs", default=50)
parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.0001)
parser.add_argument("--reg_coeff", type=float, help="regulation coefficient", default=1)

# Model configurations
# cov feature model
parser.add_argument("--cov_in_dim", type=int, help="number of covariates", default=10)
parser.add_argument("--cov_feature_dim", type=int, help="dimensionality of extracted features", default=16)
parser.add_argument("--cov_hidden_dims", type=int, nargs='+', 
                    help="List of hidden dimensions in the cov feature model", 
                    default=[64, 32])
parser.add_argument("--cov_dropout_rate", type=float, 
                    help="dropout rate in the cov feature model", 
                    default=0.5)

# treatment feature model
parser.add_argument("--treatment_in_channels", type=int, help="number of bands of the imageries", default=7)
parser.add_argument("--treatment_feature_dim", type=int, help="dimensionality of extracted features", default=16)
parser.add_argument("--treatment_feature_model", type=str, 
                    help="type the treatment feature model", 
                    default='VanillaCNN')
parser.add_argument("--treatment_dropout_rate", type=float, 
                    help="dropout rate in the treatment mapping model", 
                    default=0.5)
# outcome model
parser.add_argument("--outcome_hidden_dims", type=int, nargs='+', 
                    help="List of hidden dimensions in the outcome model", 
                    default=[64, 32, 16])
parser.add_argument("--outcome_dropout_rate", type=float, 
                    help="dropout rate in the outcome model", 
                    default=0.5)

# Parse the arguments
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

## load data
with open('data/dark_vessel_simulated_meta.pkl', 'rb') as f:
    simulated_meta = pickle.load(f)

## data partition
train_indices, val_indices, test_indices = split_indices(
    len(simulated_meta), 
    train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=args.seed)

y_f = [data['y_f'] for data in simulated_meta]
scale_mean = np.mean(y_f)
scale_std = np.std(y_f)

train_dataset = LICSDataset(
    [simulated_meta[i] for i in train_indices], scale_mean = scale_mean, scale_std = scale_std, train = True, test = False)

test_dataset_cate = LICSDataset(
    [simulated_meta[i] for i in test_indices], scale_mean = scale_mean, scale_std = scale_std, train = False, test = True)

train_dl = DataLoader(train_dataset, batch_size = args.batch_size)
test_cate_dl = DataLoader(test_dataset_cate, batch_size = args.batch_size)

covarites_feature_config = OmegaConf.create()
covarites_feature_config.in_dim = args.cov_in_dim
covarites_feature_config.feature_dim = args.cov_feature_dim
covarites_feature_config.hidden_dims = args.cov_hidden_dims
covarites_feature_config.dropout_rate = args.cov_dropout_rate

treatment_feature_config = OmegaConf.create()
treatment_feature_config.in_channels = args.treatment_in_channels
treatment_feature_config.feature_dim = args.treatment_feature_dim
treatment_feature_config.dropout_rate = args.treatment_dropout_rate

outcome_model_config = OmegaConf.create()
outcome_model_config.in_dim = covarites_feature_config.feature_dim + treatment_feature_config.feature_dim
outcome_model_config.hidden_dims = args.outcome_hidden_dims
outcome_model_config.dropout_rate = args.outcome_dropout_rate

treatment_feature_model = 'VanillaCNN'
img_ite = ImgITE(
    covarites_feature_config,
    treatment_feature_config,
    outcome_model_config,
    treatment_feature_model = treatment_feature_model,
    reg = 'HISC', reg_coeff = args.reg_coeff, learning_rate = args.learning_rate
)

print(img_ite.cov_feature)

# tracking_uri = '/home/clu/mlruns'
# mlflow.set_tracking_uri(tracking_uri)

# mlf_logger = MLFlowLogger(
#     experiment_name="LICS_Modified_GraphITE",
#     tracking_uri=tracking_uri
# )

logger = CSVLogger("LICS_Graphite" + str(args.seed), name="LICS_Graphite" + str(args.seed)) 

trainer = Trainer(max_epochs=args.max_epochs,  
                  accelerator="gpu",
                  logger=logger, enable_checkpointing=False)

# trainer.fit(img_ite, train_dl, [val_dl, val_cate_dl])
trainer.fit(img_ite, train_dl)

test_mse = trainer.test(img_ite, dataloaders = test_cate_dl)

# Prepare the data to log using the provided arguments
log_data = {
    'model': 'GraphITE',
    'seed': args.seed,
    'batch_size': args.batch_size,
    'max_epochs': args.max_epochs,
    'learning_rate': args.learning_rate,
    'reg_coeff': args.reg_coeff,
    'cov_in_dim': args.cov_in_dim,
    'cov_feature_dim': args.cov_feature_dim,
    'cov_hidden_dims': '_'.join(map(str, args.cov_hidden_dims)),
    'cov_dropout_rate': args.cov_dropout_rate,
    'treatment_feature_model': treatment_feature_model,
    'treatment_in_channels': args.treatment_in_channels,
    'treatment_feature_dim': args.treatment_feature_dim,
    'treatment_feature_model': args.treatment_feature_model,
    'outcome_hidden_dims': '_'.join(map(str, args.outcome_hidden_dims)),
    'outcome_dropout_rate': args.outcome_dropout_rate,
    'test_mse': test_mse[0]['test_mse_cate'],
}


# Define the path to the CSV file
csv_file = 'metrics/GraphITE_metric.csv'

# Check if the CSV file exists
file_exists = os.path.isfile(csv_file)

# Create a DataFrame with the log data
df = pd.DataFrame([log_data])

# Write the data to the CSV file
if file_exists:
    # Append the data if the file already exists
    df.to_csv(csv_file, mode='a', header=False, index=False)
else:
    # Create the file and write the data with headers
    df.to_csv(csv_file, mode='w', header=True, index=False)