import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)

class ImageClassifier(torch.nn.Module):
    def __init__(self, P):
        super(ImageClassifier, self).__init__()
        
        self.arch = P['arch']
        feature_extractor = torchvision.models.resnet50(pretrained=P['use_pretrained'])
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])
        

        if P['freeze_feature_extractor']:
            for param in feature_extractor.parameters():
                param.requires_grad = False
        else:
            for param in feature_extractor.parameters():
                param.requires_grad = True
        self.feature_extractor = feature_extractor
            
        self.avgpool = GlobalAvgPool2d()

        linear_classifier = torch.nn.Linear(P['feat_dim'], P['num_classes'], bias=True)
        self.linear_classifier = linear_classifier

    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def get_cam(self, x):
        feats = self.feature_extractor(x)
        CAM = F.conv2d(feats, self.linear_classifier.weight.unsqueeze(-1).unsqueeze(-1))
        return CAM

    def foward_linearinit(self, x):
        x = self.linear_classifier(x)
        return x
        
    def forward(self, x):

        feats = self.feature_extractor(x)
        pooled_feats = self.avgpool(feats)
        logits = self.linear_classifier(pooled_feats)
    
        return logits
    
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1) 
        nn.init.constant_(self.linear.weight, 0) 
        nn.init.constant_(self.linear.bias, -2)
    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

class LabelEstimator(nn.Module):
    def __init__(self, P):
        super(LabelEstimator, self).__init__()
        self.models = nn.ModuleList([LogisticRegression() for _ in range(P['num_classes'])])

    def forward(self, x):
        outputs = []
        for i in range(x.shape[1]):
            column = x[:, i].view(-1, 1) 
            output = self.models[i](column)
            outputs.append(output)
        return torch.cat(outputs, dim=1)
    
class KFunction(nn.Module):
    def __init__(self, w, b):
        super(KFunction, self).__init__()
        self.w = w
        self.b = b

    def forward(self, p):
        return 1 / (1 + torch.exp(-(self.w * p + self.b)))
class KFunction1(nn.Module):
    def __init__(self, w,b):
        super(KFunction1, self).__init__()
        self.w = w
        self.b = b

    def forward(self, x):
        numerator = (1 - self.w) * x
        denominator = 1 - self.w * x
        return (numerator / denominator) + self.b
class GaussianFunctionWithEMA(torch.nn.Module):
    def __init__(self, alpha):
        super(GaussianFunctionWithEMA, self).__init__()
        self.alpha = alpha
        mu_0,sigma_0=0.5,2
        self.register_buffer('mu_ema', torch.tensor(mu_0))
        self.register_buffer('sigma_ema', torch.tensor(sigma_0))

    def forward(self, p):
        mu_batch = p.mean()
        sigma_batch = p.std()

        # Update mu and sigma using EMA
        self.mu_ema = self.alpha * mu_batch + (1 - self.alpha) * self.mu_ema
        self.sigma_ema = self.alpha * sigma_batch + (1 - self.alpha) * self.sigma_ema

        return torch.exp(-0.5 * ((p - self.mu_ema) / self.sigma_ema) ** 2)

def VFunction(p, mu, sigma):
    return torch.exp(-0.5 * ((p - mu) / sigma) ** 2)