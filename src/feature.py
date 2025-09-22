import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import satlaspretrain_models

class CovsFeatureExtractor(nn.Module):
    def __init__(self, in_dim=5, feature_dim=128, hidden_dims=[32, 16], dropout_rate=0.2):
        super(CovsFeatureExtractor, self).__init__()
        
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
        
        # Add the output layer (from last hidden layer to feature_dim)
        layers.append(nn.Linear(hidden_dims[-1], feature_dim))
        
        # Create the sequential module
        self.network = nn.Sequential(*layers)
        
    def forward(self, cov):
        latent_features = self.network(cov)
        return latent_features


class VanillaCNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=11, feature_dim=128, dropout_rate=0.2):
        super(VanillaCNNFeatureExtractor, self).__init__()
        
        # Layer 1: 32 filters, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        
        # Layer 2: 64 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Layer 3: 128 filters, 3x3 kernel
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Fixed output size (1x1)

        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(128, feature_dim)  # No need to compute feature dim manually

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 128)

        x = self.fc1(x)
        return x

class SatlasLandSatFeatureExtractor(nn.Module):    
    def __init__(self,
                 in_channels = 11,
                 feature_dim=128,
                 backbone = 'Landsat_SwinB_SI',
                 treatment_freeze=False, 
                 learnable_impute_bands = False,
                 missing_bands = None,
                 img_size = None,
                 device = 'cuda'):

        super().__init__()

        ## If using learnable parameters as imputations
        self.learnable_impute_bands = learnable_impute_bands
        if self.learnable_impute_bands == True:
            self.missing_bands = missing_bands
            H, W = img_size
            self.missing_bands = sorted(missing_bands)  # Ensure order
            self.learnable_bands = nn.ParameterDict({
                f'band_{b}': nn.Parameter(torch.randn(1, 1, H, W)) for b in self.missing_bands
            })
        
        weights_manager = satlaspretrain_models.Weights()
        
        self.swin_model = weights_manager.get_pretrained_model(
            model_identifier = backbone, 
            device = device).backbone.backbone

        ## input layer
        if in_channels != 11 and learnable_impute_bands == False:
            self.swin_model.features[0][0] = nn.Conv2d(in_channels=in_channels, 
                                           out_channels=128, 
                                           kernel_size=4, 
                                           stride=4)
        
        ## fully-connected layer
        self.swin_model.head = nn.Linear(1024, feature_dim, bias = True)

        if treatment_freeze:
            for name, param in self.swin_model.features.named_parameters():
                if 'norm' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # unfreeze input layer
            for name, param in self.swin_model.features[0][0].named_parameters():
                param.requires_grad = True
                
    def forward(self, x):
        if self.learnable_impute_bands == True:
            B, C, H, W = x.shape
            total_bands = C + len(self.missing_bands)  # Final should be (B, 11, H, W)
    
            bands = []
            input_idx = 0
            for i in range(total_bands):  
                if i in self.missing_bands:
                    # Insert learnable parameter for missing bands
                    bands.append(self.learnable_bands[f'band_{i}'].expand(B, -1, -1, -1))
                else:
                    # Use existing bands
                    bands.append(x[:, input_idx:input_idx+1])
                    input_idx += 1
            
            x = torch.cat(bands, dim=1)

        x = self.swin_model(x)      
        
        return x


class SatlasSentinel2FeatureExtractor(nn.Module):    
    def __init__(self,
                 in_channels = 9,
                 feature_dim=128,
                 backbone= ['Sentinel2_SwinB_SI_MS', 'Sentinel2_SwinT_SI_MS'],
                 treatment_freeze=False,
                 device = 'cuda'):

        super().__init__()
        
        weights_manager = satlaspretrain_models.Weights()
        
        self.swin_model = weights_manager.get_pretrained_model(
            model_identifier = backbone, 
            device = device).backbone.backbone

        ## input layer
        if in_channels != 9:
            if backbone == 'Sentinel2_SwinB_SI_MS':
                self.swin_model.features[0][0] = nn.Conv2d(in_channels=in_channels, 
                                                           out_channels=128, 
                                                           kernel_size=4, 
                                                           stride=4)
            elif backbone == 'Sentinel2_SwinT_SI_MS':
                self.swin_model.features[0][0] = nn.Conv2d(in_channels=in_channels, 
                                                           out_channels=96, 
                                                           kernel_size=4, 
                                                           stride=4)
        
        ## fully-connected layer
        if backbone == 'Sentinel2_SwinB_SI_MS':
                self.swin_model.head = nn.Linear(1024, feature_dim, bias = True)
        elif backbone == 'Sentinel2_SwinT_SI_MS':
                self.swin_model.head = nn.Linear(768, feature_dim, bias = True) 

        if treatment_freeze:
            for name, param in self.swin_model.features.named_parameters():
                if 'norm' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # unfreeze input layer
            for name, param in self.swin_model.features[0][0].named_parameters():
                param.requires_grad = True
                
    def forward(self, x):
        x = self.swin_model(x)      
        return x


class VGGFeatureExtractor(nn.Module):
    def __init__(self, 
                 in_channels = 3, 
                 feature_dim=128, 
                 backbone = ['VGG11', 'VGG13', 'VGG16', 'VGG19']):
        super().__init__()

        if backbone == 'VGG11':
            self.vgg = torchvision.models.vgg11(pretrained=False)  # No pretrained weights
        elif backbone == 'VGG13':
            self.vgg = torchvision.models.vgg13(pretrained=False)  # No pretrained weights
        elif backbone == 'VGG16':
            self.vgg = torchvision.models.vgg16(pretrained=False)  # No pretrained weights
        elif backbone == 'VGG19':
            self.vgg = torchvision.models.vgg19(pretrained=False)  # No pretrained weights

        # Modify first conv layer to accept different channel input
        self.vgg.features[0] = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.vgg.classifier[-1] = nn.Linear(4096, feature_dim, bias = True)

    def forward(self, x):
        x = self.vgg(x)
        return x

class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, 
                 in_channels = 3, 
                 feature_dim=128, 
                 backbone = ['DenseNet121', 'DenseNet169', 'DenseNet201']):
        super().__init__()

        if backbone == 'DenseNet121':
            self.densenet = torchvision.models.densenet121(pretrained=False)  # No pretrained weights
        elif backbone == 'DenseNet169':
            self.densenet = torchvision.models.densenet169(pretrained=False)  # No pretrained weights
        elif backbone == 'DenseNet201':
            self.densenet = torchvision.models.densenet201(pretrained=False)  # No pretrained weights

        # Modify first conv layer to accept 9-channel input
        self.densenet.features.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features , feature_dim)
        
    def forward(self, x):
        x = self.densenet(x)  # Extract CNN features
        return x


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, 
                 in_channels = 3, 
                 feature_dim=128, 
                 backbone = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet152']):
        super().__init__()

        if backbone == 'ResNet18':
            resnet = torchvision.models.resnet18(pretrained=False)  # No pretrained weights
        elif backbone == 'ResNet34':
            resnet = torchvision.models.resnet34(pretrained=False)  # No pretrained weights
        elif backbone == 'ResNet50':
            resnet = torchvision.models.resnet50(pretrained=False)  # No pretrained weights
        elif backbone == 'ResNet152':
            resnet = torchvision.models.resnet152(pretrained=False)  # No pretrained weights

        # Modify first conv layer to accept 9-channel input
        resnet.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
        self.fc = nn.Linear(resnet.fc.in_features, feature_dim)  # Reduce feature dim to 128

    def forward(self, x):
        x = self.backbone(x)  # Extract CNN features
        x = torch.flatten(x, start_dim=1)  # Flatten from (batch, 512, 1, 1) to (batch, 512)
        x = self.fc(x)  # Reduce to (batch, 128)
        return x