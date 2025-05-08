# Neural Style Transfer

This repository contains an implementation of Neural Style Transfer using PyTorch, based on the seminal paper by Gatys et al.

## Overview

Neural Style Transfer is a technique that takes two images—a content image and a style image—and blends them together so that the resulting output image retains the core elements of the content image, but appears to be "painted" in the style of the style image.

This implementation is based on the original algorithm described in:

> Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2414-2423).

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- matplotlib

You can install all requirements using:
```
pip install -r requirements.txt
```

## Dataset

To use this code, you need:
- A content image (what you want to style)
- A style image (the artistic style to apply)

Place your images in the `data/` directory.

## Usage

### Quick Start

```python
from src.style_transfer import StyleTransfer
from src.config import CFG

# Initialize the model
style_transfer = StyleTransfer(
    content_path='data/content.jpg',
    style_path='data/style.jpg',
    cfg=CFG
)

# Run style transfer
stylized_image = style_transfer.run(iterations=1000)

# Save the result
style_transfer.save_image(stylized_image, 'outputs/stylized_output.jpg')
```

### Options

You can adjust parameters in `src/config.py` to customize:
- Image size
- Style weight
- Content weight
- Style layers and their weights
- Optimization settings

## Examples

Check the `notebooks/neural_style_transfer_demo.ipynb` for a complete example.

## License

MIT