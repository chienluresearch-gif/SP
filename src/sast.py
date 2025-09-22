import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import satlaspretrain_models
import pytorch_lightning as pl
from src.feature import *

class MeanOutcomeModel(nn.Module):
    def __init__(self, in_dim, hidden_dims = [64, 32], dropout_rate = 0.2):
        super(MeanOutcomeModel, self).__init__()
        
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


class SatGRD(pl.LightningModule):
    def __init__(self,
                 mean_outcome_model_config,
                 covarites_mapping_config = None,
                 treatment_mapping_config = None,
                 propensity_mapping_config = None,
                 treatment_mapping_model = ['SatlasLandSat', 'SatlasSentinel2', 'DenseNet', 'VanillaCNN', None],
                 learning_rate=1e-4, l2_lambda=10):
        super().__init__()
        
        # mean outcome model
        self.save_hyperparameters({
            "mean_outcome_model": mean_outcome_model_config,
            "training": {
                "learning_rate": learning_rate
            }
        })
        
        self.mean_outcome_model = MeanOutcomeModel(**mean_outcome_model_config)

        # representation learning models
        if covarites_mapping_config is not None:
            self.save_hyperparameters({
                "covarites_mapping": covarites_mapping_config,
            })
            self.covarites_mapping = CovsFeatureExtractor(**covarites_mapping_config)

        if treatment_mapping_config is not None:
            self.save_hyperparameters({
                "treatment_mapping": treatment_mapping_config,
            })
            if treatment_mapping_model == 'SatlasLandSat':
                self.treatment_mapping = SatlasLandSatFeatureExtractor(**treatment_mapping_config)
            elif treatment_mapping_model == 'SatlasSentinel2':
                self.treatment_mapping = SatlasSentinel2FeatureExtractor(**treatment_mapping_config)
            elif treatment_mapping_model == 'DenseNet':
                self.treatment_mapping = DenseNetFeatureExtractor(**treatment_mapping_config)
            elif treatment_mapping_model == 'VanillaCNN':
                self.treatment_mapping = VanillaCNNFeatureExtractor(**treatment_mapping_config)

        if propensity_mapping_config is not None:
            self.save_hyperparameters({
                "propensity_mapping": propensity_mapping_config,
            })
            self.propensity_mapping = CovsFeatureExtractor(**propensity_mapping_config)

        # training settings
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.criterion = nn.MSELoss()
        self.training_mean_outcome = True
        self.training_step_counter = 0
        self.automatic_optimization = False  # Important: This property activates manual optimization.
        
    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        if self.training_mean_outcome:
            if dataloader_idx  == 0:
                img, covs, outcome = batch
                # img, covs, cat, mean_ndvi, mean_ndmi, pop = batch
                mean_outcome = self.mean_outcome_model(covs)
                loss = self.criterion(mean_outcome, outcome)
                self.log('val_loss_mean_outcome', loss.item())
                return {'val_loss_mean_outcome': loss}

        else:
            if dataloader_idx  == 0:
                img, covs, outcome = batch
                # img, covs, cat, mean_ndvi, mean_ndmi, pop = batch
                mean_outcome, covariates_emb, treatment_emb, propensity_emb = self.forward(img, covs)
                y_mean_outcome = mean_outcome
                
                outcome_res = outcome - y_mean_outcome
                treatment_res = treatment_emb - propensity_emb
                loss = self.criterion((covariates_emb * treatment_res).sum(1), outcome_res)
                self.log('val_loss_emb', loss.item())
                
                return {'val_loss_emb': loss}
                
            elif dataloader_idx == 1:
                # img, cf_img, covs, pop, pop_cf, cate = batch
                img, cf_img, covs, outcome, outcome_cf, cate  = batch

                # for factual emb loss
                mean_outcome, covariates_emb, treatment_emb, propensity_emb = self.forward(img, covs)
                y_mean_outcome = mean_outcome
                
                outcome_res = outcome - y_mean_outcome
                treatment_res = treatment_emb - propensity_emb
                loss = self.criterion((covariates_emb * treatment_res).sum(1), outcome_res)
                self.log('val_factual_loss_emb', loss.item())

                # for counterfactual emb loss
                mean_outcome, covariates_emb, treatment_emb, propensity_emb = self.forward(cf_img, covs)
                y_mean_outcome = mean_outcome
                
                outcome_res = outcome_cf - y_mean_outcome
                treatment_res = treatment_emb - propensity_emb

                outcome_est = (covariates_emb * treatment_res).sum(1)
                loss_cf = self.criterion(outcome_est, outcome_res)
                self.log('val_counterfactual_loss_emb', loss_cf.item())
                
                # loss_increase = self.criterion(outcome_est[increase_label], outcome_res)
                # loss_decrease = self.criterion(outcome_est[increase_label == False], outcome_res)
                
                # self.log('val_counterfactual_increase_emb', loss_increase.item())
                # self.log('val_counterfactual_decrease_emb', loss_decrease.item())
                
                # for CATE estimation
                covariates_emb = self.covarites_mapping(covs)
                treatment_factual_emb = self.treatment_mapping(img)
                treatment_counter_factual_emb = self.treatment_mapping(cf_img)
                
                # Estimate CATE
                est_cate = (covariates_emb * (treatment_counter_factual_emb - treatment_factual_emb)).sum(1)  

                mse_cate = self.criterion(est_cate, cate)
                self.log('val_mse_cate', mse_cate.item())
                
                # mse_cate_increase = self.criterion(est_cate[increase_label], cate[increase_label])
                # self.log('val_mse_increase_cate', mse_cate_increase.item())

                # mse_cate_decrease = self.criterion(est_cate[increase_label == False], cate[increase_label == False])
                # self.log('val_mse_decrease_cate', mse_cate_decrease.item())
                
                # self.log('val_loss_cate', loss.item(), on_epoch=True)
                # self.log('val_mse_cate', mse_cate.item())
                return {'val_mse_cate': mse_cate.item()}
    
    def test_step(self, batch, batch_idx):
        # img, cf_img, covs, pop, pop_cf, cate = batch
        img, cf_img, covs, outcome, outcome_cf, cate = batch
        
        # for CATE estimation
        covariates_emb = self.covarites_mapping(covs)
        treatment_factual_emb = self.treatment_mapping(img)
        treatment_counter_factual_emb = self.treatment_mapping(cf_img)
        
        # Estimate CATE
        est_cate = (covariates_emb * (treatment_counter_factual_emb - treatment_factual_emb)).sum(1)

        # Compute MSE
        mse_cate = self.criterion(est_cate, cate)

        self.log('test_mse_cate', mse_cate.item())
        return {'test_mse_cate': mse_cate.item()}
            
    def configure_optimizers(self):
        if self.training_mean_outcome:
            opt_mom = torch.optim.Adam(self.mean_outcome_model.parameters(), lr=self.learning_rate)
            return opt_mom
        else:
            opt_nm = torch.optim.Adam(list(self.covarites_mapping.parameters()) + list(self.treatment_mapping.parameters()), lr=self.learning_rate)
            opt_prop = torch.optim.Adam(self.propensity_mapping.parameters(), lr=self.learning_rate)
            return opt_nm, opt_prop


    def switch_to_embedding_training(self):
        print('Freezing parameters in the mean outcome model, start training embedding models..')
        self.training_mean_outcome = False
        # Freeze the parameters of the mean outcome model
        for param in self.mean_outcome_model.parameters():
            param.requires_grad = False
    
    def forward(self, img, covs):
        if self.training_mean_outcome:
            return self.mean_outcome_model(covs)
        else:
            mean_outcome = self.mean_outcome_model(covs)
            covariates_emb = self.covarites_mapping(covs)
            treatment_emb = self.treatment_mapping(img)
            propensity_emb = self.propensity_mapping(covs)
            
            return mean_outcome, covariates_emb, treatment_emb, propensity_emb

    def switch_to_embedding_training(self):
        print('Freezing parameters in the mean outcome model, start training embedding models..')
        self.training_mean_outcome = False
        # Freeze the parameters of the mean outcome model
        for param in self.mean_outcome_model.parameters():
            param.requires_grad = False

    def on_train_start(self):
        """Log max_epochs from Trainer and batch_size from DataLoader"""
        batch_size = self.trainer.train_dataloader.batch_size  # Retrieve from train_dl   

        self.logger.log_hyperparams({
            "training/max_epochs": self.trainer.max_epochs,
            "training/batch_size": batch_size
        })
    
    def training_step(self, batch, batch_idx):
        img, covs, outcome = batch
        # img, covs, cat, mean_ndvi, mean_ndmi, pop = batch
        if self.training_mean_outcome:
            opt_mom = self.optimizers()
            mean_outcome = self.mean_outcome_model(covs)
            
            pred_loss = self.criterion(mean_outcome, outcome)
            # reg_l2_loss = sum((p**2).sum() for p in list(self.mean_outcome_model.parameters()))
            # loss = pred_loss + self.l2_lambda * reg_l2_loss
            loss = pred_loss
            
            self.log('train_loss_mean_outcome', pred_loss.item(), on_epoch=True)
            # self.log('train_reg_loss_mean_outcome', reg_l2_loss.item(), on_epoch=True)

            opt_mom.zero_grad()
            self.manual_backward(loss)
            opt_mom.step()
        
        else:
            opt_nm, opt_prop = self.optimizers()
            mean_outcome, covariates_emb, treatment_emb, propensity_emb = self.forward(img, covs)
            
            outcome_res = outcome - mean_outcome
            treatment_res = treatment_emb - propensity_emb
            
            pred_loss = self.criterion((covariates_emb * treatment_res).sum(1), outcome_res)
            reg_l2_loss = sum((p**2).sum() for p in list(self.covarites_mapping.parameters()) + list(self.treatment_mapping.parameters()) + list(self.propensity_mapping.parameters()))
            loss = pred_loss + self.l2_lambda * reg_l2_loss

            self.log('train_pred_loss_embedding', pred_loss.item(), on_epoch=True)
            self.log('train_reg_loss_embedding', reg_l2_loss.item(), on_epoch=True)

            if self.training_step_counter % 5 == 0:            
                opt_prop.zero_grad()
                self.manual_backward(loss)
                opt_prop.step()
            else:
                opt_nm.zero_grad()
                self.manual_backward(loss)
                opt_nm.step()

            self.training_step_counter += 1