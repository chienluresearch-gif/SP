import pickle, argparse
import os, sys
sys.path.append(os.getcwd())

from omegaconf import OmegaConf
from src.sast import *
from src.data import *
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

# Create the parser
parser = argparse.ArgumentParser(description="args for mean outcome model")

# Add arguments
parser.add_argument("--seed", type=int, help="seed for experiments", default=64)
parser.add_argument("--batch_size", type=int, help="seed for experiments", default=64)
parser.add_argument("--max_epochs", type=int, help="training epochs", default=50)
parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.0001)
parser.add_argument("--trained_mom_dir", type=str, default="train_mom_eurosat",
                    help="dir for trained mom models")


# mean outcome model
parser.add_argument("--mom_in_dims", type=int, 
                    help="List of hidden dimensions in the mean outcome model", 
                    default=7)
parser.add_argument("--mom_hidden_dims", type=int, nargs='+', 
                    help="List of hidden dimensions in the mean outcome model", 
                    default=[64, 32])
parser.add_argument("--mom_dropout_rate", type=float, 
                    help="dropout rate in the mean outcome model", 
                    default=0.2)

# Parse the arguments
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

## load data
with open('data/simulated_meta_EuroSAT.pkl', 'rb') as f:
    simulated_meta = pickle.load(f)

## data partition
train_indices, val_indices, test_indices = split_indices(
    len(simulated_meta), 
    train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=args.seed)

P_f = [data['P'] for data in simulated_meta]
scale_mean = np.mean(P_f)
scale_std = np.std(P_f)

train_dataset = EuroSATDataset(
    [simulated_meta[i] for i in train_indices], scale_mean = scale_mean, scale_std = scale_std, train = True, test = False)

train_mom_dl = DataLoader(train_dataset, batch_size = args.batch_size)

mean_outcome_model_config = OmegaConf.create()
mean_outcome_model_config.in_dim = args.mom_in_dims
mean_outcome_model_config.hidden_dims = args.mom_hidden_dims
mean_outcome_model_config.dropout_rate = args.mom_dropout_rate

mean_outcome_model = SatGRD(mean_outcome_model_config, learning_rate = args.learning_rate)

logger = CSVLogger("EuroSAT_MoM", name="EuroSAT_MoM" + str(args.seed)) 

if torch.cuda.device_count() > 1:
    trainer_1 = Trainer(max_epochs=args.max_epochs, 
                        accelerator="gpu", strategy="ddp",
                        devices="auto", logger=logger, enable_checkpointing=False)
else:
    trainer_1 = Trainer(max_epochs=args.max_epochs,  
                        accelerator="gpu",
                        logger=logger, enable_checkpointing=False)

trainer_1.fit(mean_outcome_model, train_mom_dl)

os.makedirs(args.trained_mom_dir, exist_ok=True)

trained_mom_path = args.trained_mom_dir + '/' + "trained_mom_eurosat_seed_" + str(args.seed) + "_batch_size_" + str(args.batch_size) + "max_epochs" + str(args.max_epochs) + "learning_rate" + str(args.learning_rate) +  "mom_in_dims" + str(args.mom_in_dims) + "_hidden_dims_" + str(args.mom_hidden_dims) + "_drop_rate_" + str(args.mom_dropout_rate) + ".pth"

torch.save(mean_outcome_model.mean_outcome_model.state_dict(), trained_mom_path)