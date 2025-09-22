import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from src.reg import *
from src.feature import *

class OutcomeModel(nn.Module):
    def __init__(self, in_dim, hidden_dims = [64, 32], dropout_rate = 0.2):
        super(OutcomeModel, self).__init__()
        
        layers = []
        # Add the first layer from in_dim to the first hidden layer
        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
        
        # Add hidden layers dynamically based on hidden_dims list
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
        
        # Add the output layer (from last hidden layer to out_dim)
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        # Create the sequential module
        self.network = nn.Sequential(*layers)
        
    
    def forward(self, cov):
        y = self.network(cov)
        return y


class ImgITE(pl.LightningModule):
    def __init__(self, 
                 covarites_feature_config,
                 treatment_feature_config, 
                 outcome_model_config,
                 treatment_feature_model = ['VanillaCNN','VGG', 'ResNet', 'DenseNet', 'SatlasSentinel2', 'SatlasLandSat'], 
                 reg = ['MMD', 'HISC', None], 
                 reg_coeff = 10, 
                 learning_rate = 0.0001, logging = False):
        super().__init__()

        # Save all hyperparameters automatically
        self.logging = logging

        if self.logging == True:
            self.save_hyperparameters({
                "covariates": covarites_feature_config,
                "treatment": treatment_feature_config,
                "outcome": outcome_model_config,
                "treatment_feature_model": treatment_feature_model,
                "reg": reg,
                "reg_coeff": reg_coeff,
                "training": {
                    "learning_rate": learning_rate
                }
            })
        
        self.cov_feature = CovsFeatureExtractor(**covarites_feature_config)

        if treatment_feature_model == 'VanillaCNN':
            self.treatment_feature = VanillaCNNFeatureExtractor(**treatment_feature_config)
        elif treatment_feature_model == 'VGG':
            self.treatment_feature = VGGFeatureExtractor(**treatment_feature_config)
        elif treatment_feature_model == 'ResNet':
            self.treatment_feature = ResNetFeatureExtractor(**treatment_feature_config)
        elif treatment_feature_model == 'DenseNet':
            self.treatment_feature = DenseNetFeatureExtractor(**treatment_feature_config)
        elif treatment_feature_model == 'SatlasSentinel2':
            self.treatment_feature = SatlasSentinel2FeatureExtractor(**treatment_feature_config) 
        elif treatment_feature_model == 'SatlasLandSat':
            self.treatment_feature = SatlasLandSatFeatureExtractor(**treatment_feature_config)

        self.outcome_model = OutcomeModel(**outcome_model_config)
        
        self.reg = reg
        self.reg_coeff = reg_coeff
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, imgs, covs):
        cov_features = self.cov_feature(covs)
        treatment_features = self.treatment_feature(imgs)
        outcome_model_input = torch.cat([treatment_features, cov_features], dim=1)
        
        pred_outcome = self.outcome_model(outcome_model_input)
        
        return pred_outcome, cov_features, treatment_features

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        if dataloader_idx  == 0:
            imgs, covs, outcome = batch
            
            pred_outcome, cov_features, treatment_features = self.forward(imgs, covs)
            pred_loss = self.criterion(pred_outcome, outcome)

            if self.reg is None:
                reg_loss = 0
                total_loss = pred_loss
            elif self.reg == 'MMD':
                reg_loss = mmd(treatment_features, cov_features)
                total_loss = pred_loss + self.reg_coeff * reg_loss
            elif self.reg == 'HISC':
                reg_loss = hsic_regular(treatment_features, cov_features)
                total_loss = pred_loss + self.reg_coeff * reg_loss
                
            if self.trainer is not None:
                self.log('val_pred_loss', pred_loss, on_epoch=True, sync_dist=True)
                self.log('val_reg_loss', reg_loss, on_epoch=True, sync_dist=True)
            
            return {'val_loss': total_loss}
        
        elif dataloader_idx == 1:
            f_imgs, cf_imgs, covs, outcome_f, outcome_cf, cate = batch

            f_treatment_features = self.treatment_feature(f_imgs)
            cf_treatment_features = self.treatment_feature(cf_imgs)
    
            cov_features = self.cov_feature(covs)
            
            f_outcome_model_input = torch.cat([f_treatment_features, cov_features], dim=1)
            cf_outcome_model_input = torch.cat([cf_treatment_features, cov_features], dim=1)
    
            f_pred_outcome = self.outcome_model(f_outcome_model_input)
            loss_f = self.criterion(f_pred_outcome, outcome_f)
            
            cf_pred_outcome = self.outcome_model(cf_outcome_model_input)
            loss_cf = self.criterion(cf_pred_outcome, outcome_cf)
            
            cate_est = cf_pred_outcome - f_pred_outcome
    
            mse_cate = self.criterion(cate_est, cate)
            
            if self.logging == True and self.trainer is not None:
                self.log('val_factual_loss', loss_f, on_epoch=True, sync_dist=True)
                self.log('val_counterfactual_loss', loss_cf, on_epoch=True, sync_dist=True)
                self.log('val_cate', cate.mean(), on_epoch=True, sync_dist=True)
                self.log('val_cate_est', cate_est.mean(), on_epoch=True, sync_dist=True)
                self.log('val_mse_cate', mse_cate.mean(), on_epoch=True, sync_dist=True)
                
            return {'val_mse_cate': mse_cate}

    def on_train_start(self):
        """Log max_epochs from Trainer and batch_size from DataLoader"""
        batch_size = self.trainer.train_dataloader.batch_size  # Retrieve from train_dl

        self.logger.log_hyperparams({
            "training/max_epochs": self.trainer.max_epochs,
            "training/batch_size": batch_size
        })

    def training_step(self, batch, batch_idx):
        imgs, covs, outcome = batch
            
        pred_outcome, cov_features, treatment_features = self.forward(imgs, covs)
        pred_loss = self.criterion(pred_outcome, outcome)

        if self.reg is None:
            reg_loss = 0
            total_loss = pred_loss
        elif self.reg == 'MMD':
            reg_loss = mmd(treatment_features, cov_features)
            total_loss = pred_loss + self.reg_coeff * reg_loss
        elif self.reg == 'HISC':
            reg_loss = hsic_regular(treatment_features, cov_features)
            total_loss = pred_loss + self.reg_coeff * reg_loss

        if self.logging == True and self.trainer is not None:
            self.log('pred_loss', pred_loss, on_epoch=True, sync_dist=True)
            self.log('reg_loss', reg_loss, on_epoch=True, sync_dist=True)
            self.log('total_loss', total_loss, on_epoch=True, sync_dist=True)
        
        return total_loss
    

    def test_step(self, batch, batch_idx):
        f_imgs, cf_imgs, covs, outcome, outcome_cf, cate = batch

        f_treatment_features = self.treatment_feature(f_imgs)
        cf_treatment_features = self.treatment_feature(cf_imgs)

        cov_features = self.cov_feature(covs)
        
        f_outcome_model_input = torch.cat([f_treatment_features, cov_features], dim=1)
        cf_outcome_model_input = torch.cat([cf_treatment_features, cov_features], dim=1)

        f_pred_outcome = self.outcome_model(f_outcome_model_input)
        loss_f = self.criterion(f_pred_outcome, outcome)
        
        cf_pred_outcome = self.outcome_model(cf_outcome_model_input)
        loss_cf = self.criterion(cf_pred_outcome, outcome_cf)
        
        cate_est = cf_pred_outcome - f_pred_outcome

        mse_cate = self.criterion(cate_est, cate)

        if self.trainer is not None:
            self.log('test_loss_f', loss_f, on_epoch=True, sync_dist=True)
            self.log('test_loss_cf', loss_cf, on_epoch=True, sync_dist=True)
            self.log('test_cate', cate.mean(), on_epoch=True, sync_dist=True)
            self.log('test_cate_est', cate_est.mean(), on_epoch=True, sync_dist=True)
            self.log('test_mse_cate', mse_cate.mean(), on_epoch=True, sync_dist=True)
                    
        return {'test_mse_cate': mse_cate}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer