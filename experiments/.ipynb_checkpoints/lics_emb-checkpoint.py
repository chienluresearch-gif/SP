import pickle, argparse

import os, sys
sys.path.append(os.getcwd())

import pandas as pd

from omegaconf import OmegaConf
from src.sast import *
from src.data import *
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import torch.distributed as dist  # Import for distributed training
from pytorch_lightning.loggers import CSVLogger

# Create the parser
parser = argparse.ArgumentParser(description="args for concatenate baseline")

# Add arguments
parser.add_argument("--seed", type=int, help="seed for experiments", default=64)
parser.add_argument("--batch_size", type=int, help="batch size", default=32)
parser.add_argument("--max_epochs", type=int, help="training epochs", default=50)
parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.0001)
parser.add_argument("--trained_mom_dir", type=str, default='train_mom_lics',
                    help="dir for trained mom models")

# feature dimension
parser.add_argument("--feature_dim", type=int, 
                    help="feature dimensions for learned representations", 
                    default=16)

parser.add_argument("--l2_lambda", type=float, 
                    help="L2 regularization", 
                    default=0.01)

# mean outcome model
parser.add_argument("--mom_in_dims", type=int, 
                    help="List of hidden dimensions in the mean outcome model", 
                    default=10)
parser.add_argument("--mom_hidden_dims", type=int, nargs='+', 
                    help="List of hidden dimensions in the mean outcome model", 
                    default=[128, 64, 32])
parser.add_argument("--mom_dropout_rate", type=float, 
                    help="dropout rate in the mean outcome model", 
                    default=0.5)

# covarites/ propensity mapping model
parser.add_argument("--cov_in_dims", type=int, 
                    help="List of hidden dimensions in the covarites mapping model", 
                    default=10)
parser.add_argument("--cov_hidden_dims", type=int, nargs='+', 
                    help="List of hidden dimensions in the covarites mapping model", 
                    default=[128, 64, 32])
parser.add_argument("--cov_dropout_rate", type=float, 
                    help="dropout rate in the covarites mapping model", 
                    default=0.5)

# treatment mapping model
parser.add_argument("--treatment_in_channels", type=int, 
                    help="number of bands of the imageries", default=7)
parser.add_argument("--treatment_feature_model", type=str, 
                    help="type the treatment feature model", 
                    default='SatlasLandSat')
parser.add_argument("--treatment_freeze", action="store_true", 
                    help="frozen fine-tune")
parser.add_argument("--treatment_dropout_rate", type=float, 
                    help="dropout rate in the treatment mapping model", 
                    default=0.5)
parser.add_argument("--treatment_backbone", type=str, 
                    help="backbone model", 
                    default='Landsat_SwinB_SI')

# Parse the arguments
args = parser.parse_args()

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

## load data
with open('data/dark_vessel_simulated_meta.pkl', 'rb') as f:
    simulated_meta = pickle.load(f)

## data partition
train_indices, val_indices, test_indices = split_indices(
    len(simulated_meta), 
    train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=args.seed)

## create datasets and data loaders
y_f = [data['y_f'] for data in simulated_meta]
scale_mean = np.mean(y_f)
scale_std = np.std(y_f)

train_dataset = LICSDataset(
    [simulated_meta[i] for i in train_indices], scale_mean = scale_mean, scale_std = scale_std, train = True, test = False)

test_dataset_cate = LICSDataset(
    [simulated_meta[i] for i in test_indices], scale_mean = scale_mean, scale_std = scale_std, train = False, test = True)

train_emb_dl = DataLoader(train_dataset, batch_size = args.batch_size)

test_cate_dl = DataLoader(test_dataset_cate, batch_size = args.batch_size)

## mean outcome model
mean_outcome_model_config = OmegaConf.create()
mean_outcome_model_config.in_dim = args.mom_in_dims
mean_outcome_model_config.hidden_dims = args.mom_hidden_dims
mean_outcome_model_config.dropout_rate = args.mom_dropout_rate

trained_mom_path = args.trained_mom_dir + '/' + "trained_mom_lics_seed_" + str(args.seed) + "_batch_size_" + str(args.batch_size) + "max_epochs" + str(args.max_epochs) + "learning_rate" + str(args.learning_rate) +  "mom_in_dims" + str(args.mom_in_dims) + "_hidden_dims_" + str(args.mom_hidden_dims) + "_drop_rate_" + str(args.mom_dropout_rate) + ".pth"

## covariates mapping model
covarites_mapping_config = OmegaConf.create()
covarites_mapping_config.in_dim=args.cov_in_dims
covarites_mapping_config.hidden_dims=args.cov_hidden_dims
covarites_mapping_config.feature_dim=args.feature_dim
covarites_mapping_config.dropout_rate=args.cov_dropout_rate

## propensity mapping model
propensity_mapping_config = OmegaConf.create()
propensity_mapping_config.in_dim=args.cov_in_dims
propensity_mapping_config.hidden_dims=args.cov_hidden_dims
propensity_mapping_config.feature_dim=args.feature_dim
propensity_mapping_config.dropout_rate=args.cov_dropout_rate

## treatment mapping model
treatment_mapping_config = OmegaConf.create()
treatment_mapping_config.in_channels = args.treatment_in_channels
treatment_mapping_config.feature_dim = args.feature_dim

if args.treatment_feature_model == 'VanillaCNN':
    treatment_mapping_config.dropout_rate = args.treatment_dropout_rate

if args.treatment_feature_model != 'VanillaCNN':
    treatment_mapping_config.backbone = args.treatment_backbone

if args.treatment_feature_model in ['SatlasLandSat', 'SatlasSentinel2']:
    treatment_mapping_config.treatment_freeze = args.treatment_freeze

emb_model = SatGRD(
    mean_outcome_model_config = mean_outcome_model_config, 
    covarites_mapping_config = covarites_mapping_config, 
    treatment_mapping_config = treatment_mapping_config,
    propensity_mapping_config = propensity_mapping_config,
    treatment_mapping_model = args.treatment_feature_model, 
    l2_lambda = args.l2_lambda)

emb_model.mean_outcome_model.load_state_dict(torch.load(trained_mom_path))

emb_model.switch_to_embedding_training()

# tracking_uri = '/home/clu/mlruns'
# mlflow.set_tracking_uri(tracking_uri)

# mlf_logger = MLFlowLogger(
#     experiment_name="sat_emb_LICS" + args.treatment_feature_model,
#     tracking_uri=tracking_uri
# )

logger = CSVLogger("LICS_emb" + str(args.seed), name= args.treatment_feature_model + '_' + str(args.seed)) 

if torch.cuda.device_count() > 1:
    trainer_1 = Trainer(max_epochs=args.max_epochs, 
                        accelerator="gpu", strategy="ddp",
                        devices="auto", logger=logger, enable_checkpointing=False)
else:
    trainer_1 = Trainer(max_epochs=args.max_epochs,  
                        accelerator="gpu",
                        logger=logger, enable_checkpointing=False)

# trainer_1.fit(emb_model, train_emb_dl, [val_emb_dl, val_cate_dl])
trainer_1.fit(emb_model, train_emb_dl)

test_mse = trainer_1.test(emb_model, dataloaders = test_cate_dl)

if args.treatment_freeze == True:
    log_data = {
        'model': 'SaST',
        'seed': args.seed,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'learning_rate': args.learning_rate,
        'mom_in_dims': args.mom_in_dims,
        'mom_hidden_dims': '_'.join(map(str, args.mom_hidden_dims)),
        'mom_dropout_rate': args.mom_dropout_rate,
        'feature_dim': args.feature_dim,
        'cov_hidden_dims': '_'.join(map(str, args.cov_hidden_dims)),
        'cov_dropout_rate': args.cov_dropout_rate,
        'treatment_feature_model': args.treatment_feature_model + '_freeze',
        'treatment_in_channels': args.treatment_in_channels,
        'test_mse': test_mse[0]['test_mse_cate']
    }
else:
    log_data = {
        'model': 'SaST',
        'seed': args.seed,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'learning_rate': args.learning_rate,
        'mom_in_dims': args.mom_in_dims,
        'mom_hidden_dims': '_'.join(map(str, args.mom_hidden_dims)),
        'mom_dropout_rate': args.mom_dropout_rate,
        'feature_dim': args.feature_dim,
        'cov_hidden_dims': '_'.join(map(str, args.cov_hidden_dims)),
        'cov_dropout_rate': args.cov_dropout_rate,
        'treatment_feature_model': args.treatment_feature_model,
        'treatment_in_channels': args.treatment_in_channels,
        'test_mse': test_mse[0]['test_mse_cate']
    }


# Define the path to the CSV file
csv_file = 'metrics/SaST_metric.csv'

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