"""Streamlit app for Cityscapes semantic segmentation inference."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cityseg.config import load_config
from src.cityseg.constants import CITYSCAPES_CLASSES, CITYSCAPES_PALETTE
from src.cityseg.data.label_mapping import decode_train_ids_to_colors
from src.cityseg.data.transforms import build_transforms
from src.cityseg.models.factory import create_model
from src.cityseg.training.train import resolve_device

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/default.yaml"
DEFAULT_CHECKPOINT_PATH = "outputs/train/demo_train/checkpoints/best.pt"
ARCHITECTURES = (
    "tiny_unet",
    "unet",
    "unetplusplus",
    "fpn",
    "pspnet",
    "deeplabv3plus",
    "segformer",
)
ENCODERS = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "efficientnet-b3",
    "mit_b1",
)


def main() -> None:
    st.set_page_config(page_title="Cityscapes Segmentation", layout="wide")
    st.title("Cityscapes Semantic Segmentation")
    st.write(
        "Upload a street-scene image, load a trained checkpoint, and inspect the predicted "
        "semantic mask and overlay. The app only needs the checkpoint and configuration."
    )

    settings = render_sidebar()
    uploaded_file = st.file_uploader("Upload an RGB image", type=("png", "jpg", "jpeg"))
    if uploaded_file is None:
        st.info("Upload an image to run inference.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.caption(f"Uploaded image: {image.width}x{image.height}px")

    checkpoint_path = resolve_project_path(settings["checkpoint_path"])
    if not checkpoint_path.exists():
        st.warning("Checkpoint path does not exist yet. Train a model first or adjust the path.")
        st.image(image, caption="Original image", use_container_width=True)
        return

    try:
        model, config, device = load_model_resource(
            checkpoint_path=str(checkpoint_path),
            architecture=settings["architecture"],
            encoder_name=settings["encoder_name"],
            resize_height=settings["resize_height"],
            resize_width=settings["resize_width"],
            use_checkpoint_model_config=settings["use_checkpoint_model_config"],
        )
        prediction = predict_image(model=model, image=image, config=config, device=device)
    except Exception as error:
        st.error(f"Inference failed: {error}")
        return

    mask_rgb = decode_train_ids_to_colors(prediction)
    overlay = create_overlay(image, mask_rgb, opacity=settings["opacity"])

    original_column, mask_column, overlay_column = st.columns(3)
    original_column.image(image, caption="Original", use_container_width=True)
    mask_column.image(mask_rgb, caption="Predicted mask", use_container_width=True)
    overlay_column.image(overlay, caption="Overlay", use_container_width=True)

    if settings["show_legend"]:
        st.subheader("Class legend")
        render_legend()

    st.subheader("Top predicted classes")
    for class_name, percentage in top_classes_by_percentage(prediction):
        st.write(f"{class_name}: {percentage:.2f}%")

    mask_bytes = image_to_png_bytes(Image.fromarray(mask_rgb))
    overlay_bytes = image_to_png_bytes(Image.fromarray(overlay))
    download_column, overlay_download_column = st.columns(2)
    download_column.download_button(
        "Download predicted mask PNG",
        data=mask_bytes,
        file_name="predicted_mask.png",
        mime="image/png",
    )
    overlay_download_column.download_button(
        "Download overlay PNG",
        data=overlay_bytes,
        file_name="segmentation_overlay.png",
        mime="image/png",
    )


def render_sidebar() -> dict[str, Any]:
    st.sidebar.header("Inference settings")
    checkpoint_path = st.sidebar.text_input("Checkpoint path", DEFAULT_CHECKPOINT_PATH)
    checkpoint_summary = checkpoint_model_summary(resolve_project_path(checkpoint_path))
    use_checkpoint_model_config = st.sidebar.checkbox(
        "Use model settings from checkpoint",
        value=True,
        help="Recommended. This prevents architecture and encoder mismatches.",
    )
    if checkpoint_summary:
        st.sidebar.caption(f"Checkpoint model: {checkpoint_summary}")
    architecture = st.sidebar.selectbox("Model architecture", ARCHITECTURES, index=0)
    encoder_name = st.sidebar.selectbox("Encoder", ENCODERS, index=1)
    resize_height = st.sidebar.number_input("Image height", min_value=64, max_value=2048, value=512)
    resize_width = st.sidebar.number_input("Image width", min_value=64, max_value=4096, value=1024)
    opacity = st.sidebar.slider("Overlay opacity", min_value=0.0, max_value=1.0, value=0.45)
    show_legend = st.sidebar.checkbox("Show class legend", value=True)
    return {
        "checkpoint_path": checkpoint_path,
        "architecture": architecture,
        "encoder_name": encoder_name,
        "use_checkpoint_model_config": bool(use_checkpoint_model_config),
        "resize_height": int(resize_height),
        "resize_width": int(resize_width),
        "opacity": float(opacity),
        "show_legend": bool(show_legend),
    }


@st.cache_resource
def load_model_resource(
    checkpoint_path: str,
    architecture: str,
    encoder_name: str,
    resize_height: int,
    resize_width: int,
    use_checkpoint_model_config: bool,
) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    config = build_inference_config(
        checkpoint_path=checkpoint_path,
        architecture=architecture,
        encoder_name=encoder_name,
        resize_height=resize_height,
        resize_width=resize_width,
        use_checkpoint_model_config=use_checkpoint_model_config,
    )
    device = resolve_device(config.get("training", {}).get("device", "cuda"))
    model = create_model(config).to(device)
    checkpoint = torch_load(Path(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, device


def build_inference_config(
    checkpoint_path: str,
    architecture: str,
    encoder_name: str,
    resize_height: int,
    resize_width: int,
    use_checkpoint_model_config: bool = True,
) -> dict[str, Any]:
    checkpoint_config = _checkpoint_config(Path(checkpoint_path))
    config = load_config(DEFAULT_CONFIG_PATH)
    config = _deep_update(config, checkpoint_config)
    config.setdefault("model", {})
    if use_checkpoint_model_config and checkpoint_config.get("model"):
        config["model"]["encoder_weights"] = None
    else:
        config["model"]["architecture"] = architecture
        if architecture == "tiny_unet":
            config["model"]["encoder_name"] = None
            config["model"]["encoder_weights"] = None
        else:
            config["model"]["encoder_name"] = encoder_name
            config["model"]["encoder_weights"] = None
    config.setdefault("preprocessing", {})
    config["preprocessing"]["resize_height"] = resize_height
    config["preprocessing"]["resize_width"] = resize_width
    config["preprocessing"]["eval_augmentations"] = "resize_only"
    config.setdefault("training", {})
    config["training"]["device"] = "cuda"
    return config


def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    input_tensor = preprocess_image(image, config).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    original_size = image.size
    prediction_image = Image.fromarray(prediction)
    prediction_image = prediction_image.resize(original_size, resample=Image.Resampling.NEAREST)
    return np.asarray(prediction_image, dtype=np.uint8)


def preprocess_image(image: Image.Image, config: dict[str, Any]) -> torch.Tensor:
    transform = build_inference_transform(config)
    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    transformed = transform(image=image_array)
    transformed_image = transformed["image"]
    tensor = torch.from_numpy(transformed_image.transpose(2, 0, 1)).float()
    if transformed_image.dtype == np.uint8:
        tensor = tensor / 255.0
    return tensor


def build_inference_transform(config: dict[str, Any]) -> Any:
    preprocessing = config.get("preprocessing", {})
    image_size = (
        int(preprocessing.get("resize_height", 512)),
        int(preprocessing.get("resize_width", 1024)),
    )
    return build_transforms(
        strategy="resize_only",
        image_size=image_size,
        crop_size=None,
        normalization=preprocessing.get("normalize", "none"),
    )


def create_overlay(image: Image.Image, mask_rgb: np.ndarray, opacity: float) -> np.ndarray:
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if mask_rgb.shape[:2] != original.shape[:2]:
        mask_rgb = np.asarray(
            Image.fromarray(mask_rgb).resize(image.size, resample=Image.Resampling.NEAREST)
        )
    opacity = float(np.clip(opacity, 0.0, 1.0))
    blended = original.astype(np.float32) * (1.0 - opacity) + mask_rgb.astype(np.float32) * opacity
    return np.clip(blended, 0, 255).astype(np.uint8)


def top_classes_by_percentage(mask: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
    class_ids, counts = np.unique(mask, return_counts=True)
    total = int(counts.sum())
    if total == 0:
        return []
    rows = []
    for class_id, count in zip(class_ids.tolist(), counts.tolist(), strict=True):
        if 0 <= class_id < len(CITYSCAPES_CLASSES):
            rows.append((CITYSCAPES_CLASSES[class_id], (count / total) * 100.0))
    return sorted(rows, key=lambda item: item[1], reverse=True)[:top_k]


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def checkpoint_model_summary(checkpoint_path: Path) -> str | None:
    config = _checkpoint_config(checkpoint_path)
    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        return None
    architecture = model_config.get("architecture")
    if not architecture:
        return None
    encoder_name = model_config.get("encoder_name") or "none"
    return f"{architecture} / {encoder_name}"


def render_legend() -> None:
    columns = st.columns(3)
    for index, (class_name, color) in enumerate(zip(CITYSCAPES_CLASSES, CITYSCAPES_PALETTE)):
        red, green, blue = color
        swatch = (
            f"<span style='display:inline-block;width:14px;height:14px;"
            f"background:rgb({red},{green},{blue});margin-right:8px;"
            f"border:1px solid #888;'></span>"
        )
        columns[index % 3].markdown(f"{swatch}{class_name}", unsafe_allow_html=True)


def torch_load(path: Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _checkpoint_config(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {}
    try:
        checkpoint = torch_load(checkpoint_path, map_location=torch.device("cpu"))
    except Exception:
        return {}
    config = checkpoint.get("config", {})
    return config if isinstance(config, dict) else {}


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    main()
