import torch
import matplotlib.pyplot as plt
from PIL import Image

def load_image(img_path, cfg):
    """Load and preprocess an image
    
    Args:
        img_path: Path to the image file
        cfg: Configuration object with loader and device settings
        
    Returns:
        Preprocessed image tensor
    """
    img = Image.open(img_path)
    img = cfg.loader(img).unsqueeze(dim=0)
    return img.to(cfg.device, dtype=torch.float)

def view_image(img_tensor, title=None, cfg=None):
    """Display an image tensor
    
    Args:
        img_tensor: Image tensor to display
        title: Optional title for the displayed image
        cfg: Configuration object with unloader
    """
    img = img_tensor.cpu().clone()
    img = img.squeeze()
    
    if cfg is not None:
        img = cfg.unloader(img)
    else:
        # Use default unloader if cfg not provided
        from torchvision.transforms import ToPILImage
        img = ToPILImage()(img)
    
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pause to update the plot

def save_image(img_tensor, path, cfg=None):
    """Save an image tensor to a file
    
    Args:
        img_tensor: Image tensor to save
        path: Path where to save the image
        cfg: Configuration object with unloader
    """
    img = img_tensor.cpu().clone()
    img = img.squeeze()
    
    if cfg is not None:
        img = cfg.unloader(img)
    else:
        # Use default unloader if cfg not provided
        from torchvision.transforms import ToPILImage
        img = ToPILImage()(img)
    
    img.save(path)
    print(f"Image saved to {path}")

def get_white_noise_image(cfg):
    """Generate a random white noise image tensor
    
    Args:
        cfg: Configuration object with image size and device settings
        
    Returns:
        White noise image tensor
    """
    return torch.randn((1, 3, cfg.imsize, cfg.imsize)).to(cfg.device, dtype=torch.float)