import torch
import torchvision.transforms as transforms

class CFG:
    """Configuration class for Neural Style Transfer"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Image processing
    imsize = 512
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    unloader = transforms.ToPILImage()
    
    # Default paths - update these with your actual paths
    content_image_path = 'data/content.jpg'
    style_image_path = 'data/style.jpg'
    
    # Model parameters
    content_layer = 'conv4_2'
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    # Style vs Content balance
    content_weight = 1.0      # Alpha
    style_weight = 1e6        # Beta
    
    # Optimization parameters
    optimizer = 'adam'  # 'adam' or 'lbfgs'
    learning_rate = 1e-2
    num_iterations = 10000
    print_interval = 100