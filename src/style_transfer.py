import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import VGGFeatures
from .utils import load_image, save_image, get_white_noise_image, view_image

class StyleTransfer:
    """Neural Style Transfer implementation"""
    
    def __init__(self, content_path=None, style_path=None, cfg=None):
        """Initialize the Style Transfer model
        
        Args:
            content_path: Path to content image
            style_path: Path to style image
            cfg: Configuration object
        """
        self.cfg = cfg
        
        # Set device
        self.device = self.cfg.device
        torch.set_default_device(self.device)
        
        # Load images
        self.content_path = content_path or self.cfg.content_image_path
        self.style_path = style_path or self.cfg.style_image_path
        
        self.content_img = load_image(self.content_path, self.cfg)
        self.style_img = load_image(self.style_path, self.cfg)
        
        # Initialize VGG model for feature extraction
        self.vgg = VGGFeatures(self.device)
    
    def run(self, iterations=None, show_progress=True):
        """Run the style transfer process
        
        Args:
            iterations: Number of iterations to run
            show_progress: Whether to display progress
            
        Returns:
            The generated image tensor
        """
        if iterations is None:
            iterations = self.cfg.num_iterations
        
        # Initialize with white noise
        input_img = get_white_noise_image(self.cfg)
        input_img.requires_grad_(True)
        
        # Setup optimizer
        if self.cfg.optimizer.lower() == 'lbfgs':
            optimizer = optim.LBFGS([input_img], lr=1.0, max_iter=1)
        else:  # default to Adam
            optimizer = optim.Adam([input_img], lr=self.cfg.learning_rate)
        
        content_layer = self.cfg.content_layer
        style_layers = self.cfg.style_layers
        style_weights = self.cfg.style_weights
        
        alpha = self.cfg.content_weight
        beta = self.cfg.style_weight
        
        # Run optimization
        for iter in range(iterations):
            if self.cfg.optimizer.lower() == 'lbfgs':
                def closure():
                    optimizer.zero_grad()
                    
                    # Style loss
                    style_loss = 0
                    for i, layer in enumerate(style_layers):
                        input_feat = self.vgg.compute_feat_map(input_img, layer)
                        style_feat = self.vgg.compute_feat_map(self.style_img, layer).detach()
                        
                        input_gram = self.vgg.compute_gram_matrix(input_feat)
                        style_gram = self.vgg.compute_gram_matrix(style_feat)
                        
                        # Normalize by feature map size
                        size = input_feat.size(0) * input_feat.size(1)
                        style_loss += style_weights[i] * (F.mse_loss(input_gram, style_gram, reduction='sum') / (4 * size))
                    
                    # Content loss
                    input_content_feat = self.vgg.compute_feat_map(input_img, content_layer)
                    content_feat = self.vgg.compute_feat_map(self.content_img, content_layer).detach()
                    content_loss = 0.5 * F.mse_loss(input_content_feat, content_feat, reduction='sum')
                    
                    # Total loss
                    loss = alpha * content_loss + beta * style_loss
                    loss.backward()
                    
                    return loss
                
                # Update weights
                optimizer.step(closure)
                
                # Clamp values to valid image range
                with torch.inference_mode():
                    input_img.clamp_(0, 1)
                    
                # Show progress
                if show_progress and (iter + 1) % self.cfg.print_interval == 0:
                    with torch.no_grad():
                        loss = closure()
                    print(f'Iteration: {iter + 1} | Loss: {loss.item():.4f}')
            
            else:  # Using Adam optimizer
                # Style loss
                style_loss = 0
                for i, layer in enumerate(style_layers):
                    input_feat = self.vgg.compute_feat_map(input_img, layer)
                    style_feat = self.vgg.compute_feat_map(self.style_img, layer).detach()
                    
                    input_gram = self.vgg.compute_gram_matrix(input_feat)
                    style_gram = self.vgg.compute_gram_matrix(style_feat)
                    
                    # Normalize by feature map size
                    size = input_feat.size(0) * input_feat.size(1)
                    style_loss += style_weights[i] * (F.mse_loss(input_gram, style_gram, reduction='sum') / (4 * size))
                
                # Content loss
                input_content_feat = self.vgg.compute_feat_map(input_img, content_layer)
                content_feat = self.vgg.compute_feat_map(self.content_img, content_layer).detach()
                content_loss = 0.5 * F.mse_loss(input_content_feat, content_feat, reduction='sum')
                
                # Total loss
                loss = alpha * content_loss + beta * style_loss
                
                # Update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Clamp values to valid image range
                with torch.inference_mode():
                    input_img.clamp_(0, 1)
                
                # Show progress
                if show_progress and (iter + 1) % self.cfg.print_interval == 0:
                    print(f'Iteration: {iter + 1} | Loss: {loss.item():.4f}')
        
        return input_img
    
    def display_results(self, output_img=None):
        """Display the input and output images
        
        Args:
            output_img: Generated style transfer image
        """
        if output_img is None:
            output_img = self.run(show_progress=False)
            
        # Display all three images
        view_image(output_img, title='Style-Transferred Image', cfg=self.cfg)
        view_image(self.content_img, title='Content Image', cfg=self.cfg)
        view_image(self.style_img, title='Style Image', cfg=self.cfg)
    
    def save_image(self, img_tensor, path):
        """Save the image tensor to a file
        
        Args:
            img_tensor: Image tensor to save
            path: Path to save the image
        """
        save_image(img_tensor, path, self.cfg)