import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGGFeatures:
    """VGG19 model for feature extraction in Neural Style Transfer"""
    
    def __init__(self, device=None):
        """Initialize the VGG19 model for feature extraction
        
        Args:
            device: torch device to use
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Load pre-trained VGG19 model
        self.model = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(self.device)
        
        # Feature map indices for specific layers
        self.feat_map_indices = {
            'conv1_1': 0,
            'conv1_2': 2,
            'conv2_1': 5,
            'conv2_2': 7,
            'conv3_1': 10,
            'conv3_2': 12,
            'conv3_3': 14,
            'conv3_4': 16,
            'conv4_1': 19,
            'conv4_2': 21,
            'conv4_3': 23,
            'conv4_4': 25,
            'conv5_1': 28,
            'conv5_2': 30,
            'conv5_3': 32,
            'conv5_4': 34
        }
    
    def compute_feat_map(self, image_tensor, layer='conv4_2'):
        """Compute feature map for a specific VGG19 layer
        
        Args:
            image_tensor: Input tensor image
            layer: Target layer for feature extraction
            
        Returns:
            Feature map tensor
        """
        layer_idx = self.feat_map_indices[layer]
        x = image_tensor
        
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i == layer_idx:
                # Reshape to (C, H*W) format
                feat_map = x.squeeze(0).view(x.size(1), -1)
                return feat_map
                
        return None
    
    def compute_gram_matrix(self, feat_map):
        """Compute Gram Matrix from feature map
        
        Args:
            feat_map: Feature map tensor
            
        Returns:
            Gram matrix
        """
        return torch.mm(feat_map, feat_map.t())