import numpy as np
import rasterio
from torch.utils.data import Dataset

def split_indices(data_length, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of ratios must be 1."
    
    # Optionally set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Generate a shuffled array of indices
    indices = np.arange(data_length)
    np.random.shuffle(indices)
    
    # Calculate the number of samples for each dataset
    train_end = int(train_ratio * data_length)
    val_end = train_end + int(val_ratio * data_length)
    
    # Split the indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    return train_indices, val_indices, test_indices

class LICSDataset(Dataset):
    def __init__(self, meta, scale_mean = 0, scale_std = 1, cat_dict = None, train = True, test = False):
        self.meta = meta
        # self.ima_dir = ima_dir
        # self.alt_ima_dir = alt_ima_dir

        self.scale_mean = scale_mean
        self.scale_std = scale_std

        self.train = train
        self.test = test
        # self.value_at = value_at
        
        print("Loaded ", len(self.meta), " datapoints.")

    def __getitem__(self, idx):
        if self.train:
            data = self.meta[idx]

            # img
            img_path = data['f_img_path']
            img = np.load(img_path)
            img = np.clip(img[:,:,:7] * 0.0000275 - 0.2, 0, 1)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)

            covs = np.array(list(data['covs']))
            covs = covs.astype(np.float32)

            outcome = data['y_f'].astype(np.float32)

            if self.scale_mean is not None:
                outcome = (outcome-self.scale_mean)/self.scale_std
                outcome = outcome.astype(np.float32)
            
            return img, covs, outcome

        if self.test:
            data = self.meta[idx]

            # img
            img_path = data['f_img_path']
            img = np.load(img_path)
            img = np.clip(img[:,:,:7] * 0.0000275 - 0.2, 0, 1)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)

            cf_img_path = data['cf_img_path']
            cf_img = np.load(cf_img_path)
            cf_img = np.clip(cf_img[:,:,:7] * 0.0000275 - 0.2, 0, 1)
            cf_img = np.transpose(cf_img, (2, 0, 1)).astype(np.float32)
            
            covs = np.array(list(data['covs']))
            covs = covs.astype(np.float32)
            
            outcome = data['y_f'].astype(np.float32)
            outcome_cf = data['y_cf'].astype(np.float32)
            
            if self.scale_mean is not None:
                outcome = (outcome-self.scale_mean)/self.scale_std
                outcome = outcome.astype(np.float32)
                
                outcome_cf = (outcome_cf-self.scale_mean)/self.scale_std
                outcome_cf = outcome_cf.astype(np.float32)

            cate = outcome_cf - outcome
            
            return img, cf_img, covs, outcome, outcome_cf, cate

    def __len__(self):
        return len(self.meta)


class EuroSATDataset(Dataset):
    def __init__(self, meta, scale_mean = 0, scale_std = 1, cat_dict = None, train = True, test = False):
        self.meta = meta

        self.scale_mean = scale_mean
        self.scale_std = scale_std

        self.train = train
        self.test = test
        # self.value_at = value_at
        
        print("Loaded ", len(self.meta), " datapoints.")

    def __getitem__(self, idx):
        if self.train:
            data = self.meta[idx]

            # img
            img_path = data['f_img_path']
            with rasterio.open(img_path) as src:
                img = src.read()
            
            img = img.astype(np.float32)
            
            covs = np.array(list(data['covs']))
            covs = covs.astype(np.float32)

            outcome = data['P'].astype(np.float32)

            if self.scale_mean is not None:
                outcome = (outcome-self.scale_mean)/self.scale_std
                outcome = outcome.astype(np.float32)
            
            return img, covs, outcome

        if self.test:
            data = self.meta[idx]

            # img
            img_path = data['f_img_path']
            with rasterio.open(img_path) as src:
                img = src.read()
            
            img = img.astype(np.float32)
            
            # cf_img
            cf_img_path = data['cf_img_path']
            with rasterio.open(cf_img_path) as src:
                cf_img = src.read()
            
            cf_img = cf_img.astype(np.float32)
            
            covs = np.array(list(data['covs']))
            covs = covs.astype(np.float32)
            
            outcome = data['P'].astype(np.float32)
            outcome_cf = data['P_cf'].astype(np.float32)
            
            if self.scale_mean is not None:
                outcome = (outcome-self.scale_mean)/self.scale_std
                outcome = outcome.astype(np.float32)
                
                outcome_cf = (outcome_cf-self.scale_mean)/self.scale_std
                outcome_cf = outcome_cf.astype(np.float32)

            cate = outcome_cf - outcome
            
            return img, cf_img, covs, outcome, outcome_cf, cate

    def __len__(self):
        return len(self.meta)