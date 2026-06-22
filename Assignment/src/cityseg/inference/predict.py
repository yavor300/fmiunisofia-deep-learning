"""Inference helpers for semantic segmentation models."""

from __future__ import annotations

import numpy as np
import torch

from src.cityseg.data.label_mapping import decode_train_ids_to_colors


def predict_train_ids(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Predict a single train-ID mask from a channel-first image tensor."""
    if image_tensor.ndim == 3:
        batch = image_tensor.unsqueeze(0)
    elif image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
        batch = image_tensor
    else:
        raise ValueError("image_tensor must have shape [3, H, W] or [1, 3, H, W].")

    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(batch.to(target_device))
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    if was_training:
        model.train()
    return prediction


def predict_color_mask(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Predict a single RGB color mask from a channel-first image tensor."""
    train_ids = predict_train_ids(model=model, image_tensor=image_tensor, device=device)
    return decode_train_ids_to_colors(train_ids)
