import os
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
from PIL import Image, ImageOps
import gc

import random
from torch.utils.data import Subset


def prepare_batched_data(batch, device, dtype):
    """Recursively move tensors in a nested dict to the device and cast to weight_dtype."""
    for key, value in batch.items():
        if isinstance(value, dict):
            batch[key] = prepare_batched_data(value, device, dtype)
        elif isinstance(value, torch.Tensor):
            if value.ndim == 3:
                batch[key] = value.unsqueeze(0).to(device).to(dtype)
            else:
                batch[key] = value.to(device).to(dtype)
    return batch


def count_parameters_in_M(model: nn.Module):
    """
    Calculate the total number of parameters and trainable parameters of a PyTorch model,
    and return them in millions (M).
    
    Args:
        model (nn.Module): The PyTorch model to calculate the parameters for.
    Returns:
        tuple: Total number of parameters and trainable parameters, both in millions (M).
    """
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params_in_m = total_params / 1e6
    trainable_params_in_m = trainable_params / 1e6
    
    return total_params_in_m, trainable_params_in_m
    # print(f"{model.__class__.__name__} total params: {total_params_in_m:.2f}M, trainable params: {trainable_params_in_m:.2f}M")


def tensor2pil(tensor, width=512, height=512):
    target_size = (width, height)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('tensor must be a PyTorch tensor.')

    if tensor.ndim > 2:
        tensor = tensor.squeeze()
    
    array_img = tensor.cpu().to(dtype=torch.float32).numpy()
    
    if array_img.ndim == 3 and array_img.shape[0] in [1, 3]:
        array_img = array_img.transpose(1, 2, 0)
    
    if array_img.ndim == 2 or array_img.shape[2] == 1:
        array_img = np.expand_dims(array_img, axis=-1)
        array_img = np.repeat(array_img, 3, axis=-1)

    array_img = np.clip(array_img, 0, 1)
    
    array_img = (array_img * 255).astype(np.uint8)

    pil_image = Image.fromarray(array_img)

    pil_image = pil_image.resize(target_size, Image.LANCZOS)
    
    return pil_image


def convert_png_to_rgb_with_white_bg(input_path, pad_color=(255, 255, 255)):
    """
    Convert a PNG image with transparent background to an RGB image with white background.

    Args:
        input_path (str)
        pad_color (tuple)
    
    Returns:
        PIL.Image
    """
    image = Image.open(input_path)
    
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert("RGBA")
        white_bg = Image.new("RGB", image.size, pad_color)
        white_bg.paste(image, mask=image.split()[3])  # 3 是 alpha 通道
        final_image = white_bg
    else:
        final_image = image.convert("RGB")
    
    return final_image

def get_bounding_box(image, bg_color=(255, 255, 255)):
    """
    Get the minimum bounding box of the non-background part of the image.

    Args:
        image (PIL.Image)
        bg_color (tuple)
    
    Returns:
        tuple: (left, upper, right, lower) 
    
    """
    np_image = np.array(image)
    
    if np_image.shape[2] == 4:
        np_image = np_image[:, :, :3]
    
    mask = (np_image != bg_color).any(axis=2)
    
    if not mask.any():
        return (0, 0, image.width, image.height)
    
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1  # slices are exclusive at the top
    
    return (x_min, y_min, x_max, y_max)

def resize_with_bbox(image, target_size, pad_color=(255, 255, 255), scale=0.8):
    """
    Resize the image based on the bounding box of the non-background part and pad to the target size.

    Args:
        image (PIL.Image)
        target_size (int)
        pad_color (tuple)
        scale (float)
    
    Returns:
        PIL.Image: 
    
    """
    bbox = get_bounding_box(image, bg_color=pad_color)
    cropped = image.crop(bbox)
    cropped_width, cropped_height = cropped.size

    long_side = max(cropped_width, cropped_height)
    scale_factor = (target_size * scale) / long_side
    new_width = int(cropped_width * scale_factor)
    new_height = int(cropped_height * scale_factor)
    
    resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    final_image = Image.new("RGB", (target_size, target_size), pad_color)
    
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    
    final_image.paste(resized, (paste_x, paste_y))
    
    return final_image

def process_image(image_path, target_size, pad_color=(255, 255, 255), scale=0.8):
    """
    Complete image processing pipeline: convert background to white, resize based on bounding box, and pad to target size.

    Args:
        image_path (str): Path to the input image.
        target_size (int): Target width and height of the output image.
        pad_color (tuple): Padding color, default is white.
        scale (float): Proportion of the object in the output image, default is 0.9.
    
    Returns:
        PIL.Image: Processed image.
    
    """
    image_pil = convert_png_to_rgb_with_white_bg(image_path, pad_color=pad_color)

    image_pil = resize_with_bbox(image_pil, target_size=target_size, pad_color=pad_color, scale=scale)
    
    return image_pil