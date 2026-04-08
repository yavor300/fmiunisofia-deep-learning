from __future__ import annotations

import io
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None


VOC_CLASS_NAMES: List[str] = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class _ConvBNLeaky(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )


class _InceptionLikeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        branch1x1: int,
        branch3x3_reduce: int,
        branch3x3: int,
        branch5x5_reduce: int,
        branch5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        self.branch1 = _ConvBNLeaky(in_channels, branch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            _ConvBNLeaky(in_channels, branch3x3_reduce, kernel_size=1),
            _ConvBNLeaky(branch3x3_reduce, branch3x3, kernel_size=3),
        )

        # class materials replace heavy 5x5 with stacked 3x3
        self.branch3 = nn.Sequential(
            _ConvBNLeaky(in_channels, branch5x5_reduce, kernel_size=1),
            _ConvBNLeaky(branch5x5_reduce, branch5x5, kernel_size=3),
            _ConvBNLeaky(branch5x5, branch5x5, kernel_size=3),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            _ConvBNLeaky(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
                self.branch4(x),
            ],
            dim=1,
        )


class _ResidualBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        hidden = max(out_channels // 4, 8)
        self.conv1 = _ConvBNLeaky(in_channels, hidden, kernel_size=1, stride=1)
        self.conv2 = _ConvBNLeaky(hidden, hidden, kernel_size=3, stride=stride)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act(out + residual)
        return out


class YOLOResults:
    def __init__(self, xyxy: List[Tensor], names: Sequence[str]) -> None:
        self.xyxy = xyxy
        self.names = list(names)

    def print(self) -> None:
        for idx, prediction in enumerate(self.xyxy):
            if prediction.numel() == 0:
                print(f"image {idx}: no detections")
                continue

            classes = prediction[:, 5].to(torch.int64)
            unique_classes, counts = torch.unique(classes, return_counts=True)
            parts: List[str] = []
            for cls_idx, count in zip(unique_classes.tolist(), counts.tolist()):
                name = self.names[cls_idx] if 0 <= cls_idx < len(self.names) else str(cls_idx)
                suffix = "s" if count > 1 else ""
                parts.append(f"{count} {name}{suffix}")
            print(f"image {idx}: " + ", ".join(parts))

    def pandas(self) -> "_YOLOPandasResults":
        return _YOLOPandasResults(self.xyxy, self.names)


class _YOLOPandasResults:
    def __init__(self, xyxy: List[Tensor], names: Sequence[str]) -> None:
        self.xyxy = [self._as_pandas_or_records(pred, names) for pred in xyxy]

    @staticmethod
    def _as_pandas_or_records(prediction: Tensor, names: Sequence[str]) -> Any:
        rows: List[dict[str, Any]] = []
        for row in prediction.detach().cpu().tolist():
            cls_idx = int(row[5])
            rows.append(
                {
                    "xmin": float(row[0]),
                    "ymin": float(row[1]),
                    "xmax": float(row[2]),
                    "ymax": float(row[3]),
                    "confidence": float(row[4]),
                    "class": cls_idx,
                    "name": names[cls_idx] if 0 <= cls_idx < len(names) else str(cls_idx),
                }
            )

        try:
            return pd.DataFrame(
                rows,
                columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"],
            )
        except Exception:
            return rows


class YOLOv1(nn.Module):
    SUPPORTED_BACKBONES = ("vgg19", "googlenet", "inception_v3", "resnet50")

    def __init__(
        self,
        backbone: str = "googlenet",
        *,
        num_classes: int = 20,
        grid_size: int = 7,
        boxes_per_cell: int = 2,
        input_size: int = 448,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        class_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()

        if backbone not in self.SUPPORTED_BACKBONES:
            supported = ", ".join(self.SUPPORTED_BACKBONES)
            raise ValueError(f"Unsupported backbone '{backbone}'. Supported: {supported}")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if boxes_per_cell <= 0:
            raise ValueError("boxes_per_cell must be positive")

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        self.class_names = self._resolve_class_names(class_names, num_classes)

        backbone_module, backbone_channels = self._build_backbone(backbone)
        self.backbone = backbone_module

        # extra conv layers + linear prediction head per grid cell
        self.detection_head = nn.Sequential(
            _ConvBNLeaky(backbone_channels, 512, kernel_size=3),
            _ConvBNLeaky(512, 512, kernel_size=3),
            _ConvBNLeaky(512, 1024, kernel_size=3),
            _ConvBNLeaky(1024, 1024, kernel_size=3),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
            nn.Conv2d(1024, boxes_per_cell * 5 + num_classes, kernel_size=1),
        )

    @staticmethod
    def _resolve_class_names(class_names: Sequence[str] | None, num_classes: int) -> List[str]:
        if class_names is not None:
            names = list(class_names)
        elif num_classes == len(VOC_CLASS_NAMES):
            names = list(VOC_CLASS_NAMES)
        else:
            names = [f"class_{idx}" for idx in range(num_classes)]

        if len(names) < num_classes:
            names.extend([f"class_{idx}" for idx in range(len(names), num_classes)])
        return names[:num_classes]

    def _build_backbone(self, backbone: str) -> Tuple[nn.Module, int]:
        if backbone == "vgg19":
            return self._build_vgg19_backbone()
        if backbone == "googlenet":
            return self._build_googlenet_backbone()
        if backbone == "inception_v3":
            return self._build_inception_v3_backbone()
        if backbone == "resnet50":
            return self._build_resnet50_backbone()
        raise AssertionError("Unexpected backbone")

    @staticmethod
    def _build_vgg19_backbone() -> Tuple[nn.Module, int]:
        layers: List[nn.Module] = []
        in_channels = 3
        cfg: List[Union[int, str]] = [32, 32, "M", 64, 64, "M", 128, 128, 128, 128, "M", 256, 256, 256, 256, "M"]
        for item in cfg:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = int(item)
                layers.append(_ConvBNLeaky(in_channels, out_channels, kernel_size=3))
                in_channels = out_channels
        return nn.Sequential(*layers), in_channels

    @staticmethod
    def _build_googlenet_backbone() -> Tuple[nn.Module, int]:
        layers: List[nn.Module] = [
            _ConvBNLeaky(3, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _ConvBNLeaky(64, 64, kernel_size=1),
            _ConvBNLeaky(64, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _InceptionLikeBlock(192, 64, 96, 128, 16, 32, 32),
            _InceptionLikeBlock(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _InceptionLikeBlock(480, 192, 96, 208, 16, 48, 64),
            _InceptionLikeBlock(512, 160, 112, 224, 24, 64, 64),
        ]
        return nn.Sequential(*layers), 512

    @staticmethod
    def _build_inception_v3_backbone() -> Tuple[nn.Module, int]:
        layers: List[nn.Module] = [
            _ConvBNLeaky(3, 32, kernel_size=3, stride=2),
            _ConvBNLeaky(32, 32, kernel_size=3),
            _ConvBNLeaky(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _ConvBNLeaky(64, 80, kernel_size=1),
            _ConvBNLeaky(80, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _InceptionLikeBlock(192, 64, 48, 64, 64, 96, 32),
            _InceptionLikeBlock(256, 64, 64, 96, 64, 96, 64),
            _InceptionLikeBlock(320, 96, 64, 128, 64, 96, 64),
        ]
        return nn.Sequential(*layers), 384

    @staticmethod
    def _build_resnet50_backbone() -> Tuple[nn.Module, int]:
        def make_layer(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
            modules: List[nn.Module] = [_ResidualBottleneck(in_ch, out_ch, stride=stride)]
            for _ in range(1, blocks):
                modules.append(_ResidualBottleneck(out_ch, out_ch, stride=1))
            return nn.Sequential(*modules)

        stem = nn.Sequential(
            _ConvBNLeaky(3, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        layer1 = make_layer(64, 64, blocks=2, stride=1)
        layer2 = make_layer(64, 128, blocks=2, stride=2)
        layer3 = make_layer(128, 256, blocks=2, stride=2)
        layer4 = make_layer(256, 512, blocks=2, stride=2)

        return nn.Sequential(stem, layer1, layer2, layer3, layer4), 512

    def forward(self, x: Any) -> Union[Tensor, YOLOResults]:
        if torch.is_tensor(x):
            return self._forward_tensor(x)

        if isinstance(x, (list, tuple)):
            return self._inference_from_sources(list(x))

        # Single source convenience API.
        if isinstance(x, (str, Path)):
            return self._inference_from_sources([x])

        raise TypeError("Unsupported input type. Use a tensor batch, image source list, or a single image source")

    def _forward_tensor(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError("Input tensor must have shape [N, C, H, W] or [C, H, W]")

        x = x.to(dtype=torch.float32)
        feats = self.backbone(x)
        pred = self.detection_head(feats)
        pred = pred.permute(0, 2, 3, 1).contiguous()
        return pred

    def _inference_from_sources(self, sources: List[Any]) -> YOLOResults:
        if len(sources) == 0:
            return YOLOResults([], self.class_names)

        batch_tensors: List[Tensor] = []
        original_hw: List[Tuple[int, int]] = []

        for source in sources:
            tensor_image, hw = self._source_to_tensor(source)
            batch_tensors.append(tensor_image)
            original_hw.append(hw)

        device = next(self.parameters()).device
        batch = torch.stack(batch_tensors, dim=0).to(device=device)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            raw_pred = self._forward_tensor(batch)
            decoded = self._decode_predictions(raw_pred, original_hw)
        if was_training:
            self.train()

        return YOLOResults(decoded, self.class_names)

    def _source_to_tensor(self, source: Any) -> Tuple[Tensor, Tuple[int, int]]:
        if torch.is_tensor(source):
            tensor = source.detach().clone().to(dtype=torch.float32)
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim != 3:
                raise ValueError("Tensor image source must have shape [C, H, W] or [H, W, C]")
            if tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)

            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)

            h, w = int(tensor.shape[1]), int(tensor.shape[2])
            resized = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            resized = resized.clamp(0.0, 1.0)
            return resized, (h, w)

        if isinstance(source, Path):
            source = str(source)

        if isinstance(source, str):
            maybe_tensor = self._source_str_to_tensor(source)
            if maybe_tensor is not None:
                return maybe_tensor
            # Fallback when Pillow/network are not available in the environment.
            blank = torch.zeros(3, self.input_size, self.input_size, dtype=torch.float32)
            return blank, (self.input_size, self.input_size)

        raise TypeError(f"Unsupported source type: {type(source)!r}")

    def _source_str_to_tensor(self, source: str) -> Tuple[Tensor, Tuple[int, int]] | None:
        if Image is None:
            return None

        data: bytes | None = None
        parsed = urlparse(source)

        if parsed.scheme in {"http", "https"}:
            try:
                with urlopen(source, timeout=10) as response:
                    data = response.read()
            except Exception:
                return None
        else:
            file_path = Path(source)
            if not file_path.exists() or not file_path.is_file():
                return None
            data = file_path.read_bytes()

        try:
            with Image.open(io.BytesIO(data)) as image:
                image = image.convert("RGB")
                original_hw = (int(image.height), int(image.width))
                image = image.resize((self.input_size, self.input_size))
                pixels = torch.tensor(list(image.getdata()), dtype=torch.float32)
                pixels = pixels.view(self.input_size, self.input_size, 3) / 255.0
                tensor = pixels.permute(2, 0, 1).contiguous()
                return tensor, original_hw
        except Exception:
            return None

    def _decode_predictions(self, pred: Tensor, original_hw: Sequence[Tuple[int, int]]) -> List[Tensor]:
        batch_size, s_h, s_w, _ = pred.shape
        if s_h != self.grid_size or s_w != self.grid_size:
            raise RuntimeError("Prediction grid shape does not match configured grid_size")

        detections: List[Tensor] = []
        device = pred.device

        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.grid_size, device=device),
            torch.arange(self.grid_size, device=device),
            indexing="ij",
        )
        grid_x = grid_x.to(dtype=pred.dtype)
        grid_y = grid_y.to(dtype=pred.dtype)

        class_logits = pred[..., self.boxes_per_cell * 5 :]
        class_probs = torch.softmax(class_logits, dim=-1)
        class_conf, class_idx = class_probs.max(dim=-1)

        for batch_idx in range(batch_size):
            img_pred = pred[batch_idx]
            img_class_conf = class_conf[batch_idx]
            img_class_idx = class_idx[batch_idx]

            all_boxes: List[Tensor] = []
            all_scores: List[Tensor] = []
            all_classes: List[Tensor] = []

            for box_id in range(self.boxes_per_cell):
                start = box_id * 5
                box_pred = img_pred[..., start : start + 5]

                x = (torch.sigmoid(box_pred[..., 0]) + grid_x) / self.grid_size
                y = (torch.sigmoid(box_pred[..., 1]) + grid_y) / self.grid_size
                w = torch.sigmoid(box_pred[..., 2])
                h = torch.sigmoid(box_pred[..., 3])
                obj_conf = torch.sigmoid(box_pred[..., 4])

                score = obj_conf * img_class_conf

                xmin = (x - 0.5 * w).clamp(0.0, 1.0)
                ymin = (y - 0.5 * h).clamp(0.0, 1.0)
                xmax = (x + 0.5 * w).clamp(0.0, 1.0)
                ymax = (y + 0.5 * h).clamp(0.0, 1.0)

                boxes = torch.stack((xmin, ymin, xmax, ymax), dim=-1).reshape(-1, 4)
                scores = score.reshape(-1)
                classes = img_class_idx.reshape(-1)

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)

            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            classes = torch.cat(all_classes, dim=0)

            keep = scores >= self.conf_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]

            if boxes.numel() == 0:
                detections.append(torch.empty((0, 6), device=device, dtype=pred.dtype))
                continue

            keep_indices = self._nms(boxes, scores, self.iou_threshold, self.max_detections)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            classes = classes[keep_indices]

            h_img, w_img = original_hw[batch_idx]
            scale = torch.tensor([w_img, h_img, w_img, h_img], device=device, dtype=pred.dtype)
            boxes_abs = boxes * scale

            output = torch.cat(
                [
                    boxes_abs,
                    scores.unsqueeze(1),
                    classes.to(dtype=pred.dtype).unsqueeze(1),
                ],
                dim=1,
            )
            detections.append(output)

        return detections

    @staticmethod
    def _box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

        x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
        x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

        inter_x1 = torch.maximum(x11[:, None], x21[None, :])
        inter_y1 = torch.maximum(y11[:, None], y21[None, :])
        inter_x2 = torch.minimum(x12[:, None], x22[None, :])
        inter_y2 = torch.minimum(y12[:, None], y22[None, :])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
        area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)

        union = area1[:, None] + area2[None, :] - inter_area
        return inter_area / union.clamp(min=1e-6)

    def _nms(self, boxes: Tensor, scores: Tensor, iou_threshold: float, max_detections: int) -> Tensor:
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)

        order = torch.argsort(scores, descending=True)
        keep: List[int] = []

        while order.numel() > 0 and len(keep) < max_detections:
            i = int(order[0])
            keep.append(i)
            if order.numel() == 1:
                break

            rest = order[1:]
            iou = self._box_iou(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
            order = rest[iou <= iou_threshold]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def load(model_name: str, **kwargs: Any) -> YOLOv1:
    normalized = model_name.strip().lower()
    if normalized != "yolov1":
        raise ValueError(f"Unknown model '{model_name}'. Supported models: ['yolov1']")

    # Accepted for API compatibility. We do not ship downloaded checkpoints in this course package.
    kwargs.pop("pretrained", None)

    return YOLOv1(**kwargs)


__all__ = ["YOLOv1", "YOLOResults", "load"]
