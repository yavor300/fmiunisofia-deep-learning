"""Reusable semantic segmentation training pipeline."""

from __future__ import annotations

import argparse
import csv
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import torch
from torch import nn
from tqdm.auto import tqdm

from src.cityseg.config import load_config, prepare_run, save_resolved_config
from src.cityseg.data.dataloaders import create_train_val_dataloaders
from src.cityseg.data.label_mapping import decode_train_ids_to_colors
from src.cityseg.models.factory import create_model
from src.cityseg.reporting.build_model_report import build_model_report
from src.cityseg.training.experiment_logger import append_experiment_result
from src.cityseg.training.losses import (
    CrossEntropyDiceLoss,
    CrossEntropyLovaszLoss,
    CrossEntropyLoss,
    DiceLoss,
    FocalDiceLoss,
    FocalLoss,
)
from src.cityseg.training.metrics import SegmentationMetrics
from src.cityseg.training.schedulers import Scheduler, build_scheduler, step_scheduler

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_mean_iou",
    "val_mean_dice",
    "val_pixel_accuracy",
    "learning_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output run directory name.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    return parser.parse_args()


def train_model(
    config: dict[str, Any],
    run_name: str | None = None,
    resume: str | Path | None = None,
) -> tuple[Path, list[dict[str, float | int]]]:
    config.setdefault("runtime", {})["stage"] = "train"
    if resume is not None:
        config["runtime"]["resume"] = str(resume)
    output_dir, _ = prepare_run(config, stage="train", run_name=run_name)
    save_resolved_config(config, output_dir, filename="config.yaml")

    device = resolve_device(config.get("training", {}).get("device", "cuda"))
    train_loader, val_loader = create_train_val_dataloaders(config)
    model = create_model(config).to(device)
    loss_fn = build_loss(config.get("loss", {})).to(device)
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}))
    metrics = SegmentationMetrics(
        num_classes=int(config.get("model", {}).get("num_classes", 19)),
        ignore_index=int(config.get("loss", {}).get("ignore_index", 255)),
    )
    use_amp = bool(config.get("training", {}).get("mixed_precision", False))
    use_amp = use_amp and device.type == "cuda"
    scaler = create_grad_scaler(use_amp)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_mean_iou = float("-inf")
    history: list[dict[str, float | int]] = []
    if resume is not None:
        start_epoch, best_mean_iou, history = load_training_state(
            checkpoint_path=Path(resume),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    epochs = int(config.get("training", {}).get("epochs", 1))
    progress_enabled = bool(config.get("training", {}).get("progress_bar", True))
    patience = config.get("training", {}).get("early_stopping_patience")
    patience = int(patience) if patience is not None else None
    stale_epochs = 0

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            gradient_clip_norm=config.get("training", {}).get("gradient_clip_norm"),
            epoch=epoch,
            total_epochs=epochs,
            progress_enabled=progress_enabled,
        )
        validation = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            metrics=metrics,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=epochs,
            progress_enabled=progress_enabled,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation["val_loss"],
            "val_mean_iou": validation["mean_iou"],
            "val_mean_dice": validation["mean_dice"],
            "val_pixel_accuracy": validation["pixel_accuracy"],
            "learning_rate": current_learning_rate(optimizer),
        }
        step_scheduler(scheduler, validation_metric=float(row["val_mean_iou"]))
        history.append(row)
        is_best = float(row["val_mean_iou"]) > best_mean_iou
        if is_best:
            best_mean_iou = float(row["val_mean_iou"])
            stale_epochs = 0
        else:
            stale_epochs += 1

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_mean_iou=best_mean_iou,
            history=history,
            config=config,
        )
        if is_best:
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_mean_iou=best_mean_iou,
                history=history,
                config=config,
            )

        write_history_csv(history, output_dir / "history.csv")
        write_training_plots(history, output_dir)
        if progress_enabled:
            tqdm.write(
                f"Epoch {epoch}/{epochs} "
                f"train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} "
                f"val_mIoU={row['val_mean_iou']:.4f} "
                f"val_mDice={row['val_mean_dice']:.4f} "
                f"val_acc={row['val_pixel_accuracy']:.4f}"
            )
        if patience is not None and stale_epochs >= patience:
            break

    write_predictions_preview(model, val_loader, output_dir / "predictions_preview.png", device)
    log_training_experiment(config, output_dir, history)
    return output_dir, history


def log_training_experiment(
    config: dict[str, Any],
    output_dir: Path,
    history: list[dict[str, float | int]],
) -> None:
    if not history:
        return

    best_row = max(history, key=lambda row: float(row["val_mean_iou"]))
    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))
    results_path = reports_dir / "experiment_results.csv"
    model_report_path = reports_dir / "model_report.xlsx"
    checkpoint_path = output_dir / "checkpoints" / "best.pt"
    append_experiment_result(
        config=config,
        metrics={
            "mean_iou": best_row["val_mean_iou"],
            "mean_dice": best_row["val_mean_dice"],
            "pixel_accuracy": best_row["val_pixel_accuracy"],
        },
        results_path=results_path,
        experiment_id=output_dir.name,
        checkpoint_path=checkpoint_path,
        comments=f"Training run; best epoch {best_row['epoch']} by validation mean IoU.",
    )
    build_model_report(results_path=results_path, output_path=model_report_path)


def build_loss(config: dict[str, Any]) -> nn.Module:
    name = str(config.get("name", "cross_entropy")).lower()
    ignore_index = int(config.get("ignore_index", 255))
    if name in {"cross_entropy", "ce"}:
        return CrossEntropyLoss(ignore_index=ignore_index)
    if name == "dice":
        return DiceLoss(ignore_index=ignore_index)
    if name == "focal":
        return FocalLoss(
            gamma=float(config.get("gamma", 2.0)),
            alpha=config.get("alpha"),
            ignore_index=ignore_index,
        )
    if name in {"cross_entropy_dice", "ce_dice"}:
        return CrossEntropyDiceLoss(ignore_index=ignore_index)
    if name in {"cross_entropy_lovasz", "ce_lovasz"}:
        return CrossEntropyLovaszLoss(ignore_index=ignore_index)
    if name == "focal_dice":
        return FocalDiceLoss(
            gamma=float(config.get("gamma", 2.0)),
            alpha=config.get("alpha"),
            ignore_index=ignore_index,
        )
    raise ValueError(f"Unsupported loss: {name}")


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("name", "adamw")).lower()
    lr = float(config.get("lr", 3e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScalerLike,
    device: torch.device,
    use_amp: bool,
    gradient_clip_norm: Any,
    epoch: int | None = None,
    total_epochs: int | None = None,
    progress_enabled: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(
        dataloader,
        desc=_progress_description("train", epoch, total_epochs),
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not progress_enabled,
    )
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            logits = model(images)
            loss = loss_fn(logits, masks)
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")
    return total_loss / max(total_samples, 1)


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    metrics: SegmentationMetrics,
    device: torch.device,
    use_amp: bool,
    epoch: int | None = None,
    total_epochs: int | None = None,
    progress_enabled: bool = True,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_sums = {"mean_iou": 0.0, "mean_dice": 0.0, "pixel_accuracy": 0.0}
    with torch.no_grad():
        progress = tqdm(
            dataloader,
            desc=_progress_description("val", epoch, total_epochs),
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not progress_enabled,
        )
        for images, masks in progress:
            images = images.to(device)
            masks = masks.to(device)
            with autocast_context(use_amp):
                logits = model(images)
                loss = loss_fn(logits, masks)
            batch_metrics = metrics(logits, masks)
            batch_size = images.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            for key in metric_sums:
                metric_sums[key] += float(batch_metrics[key].detach().cpu()) * batch_size
            progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

    denominator = max(total_samples, 1)
    return {
        "val_loss": total_loss / denominator,
        "mean_iou": metric_sums["mean_iou"] / denominator,
        "mean_dice": metric_sums["mean_dice"] / denominator,
        "pixel_accuracy": metric_sums["pixel_accuracy"] / denominator,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Scheduler | None,
    epoch: int,
    best_mean_iou: float,
    history: list[dict[str, float | int]],
    config: dict[str, Any],
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_mean_iou": best_mean_iou,
        "history": history,
        "config": config,
    }
    torch.save(state, path)


def load_training_state(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Scheduler | None,
    device: torch.device,
) -> tuple[int, float, list[dict[str, float | int]]]:
    checkpoint = _torch_load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_mean_iou = float(checkpoint.get("best_mean_iou", float("-inf")))
    history = list(checkpoint.get("history", []))
    return start_epoch, best_mean_iou, history


def write_history_csv(history: list[dict[str, float | int]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row[field] for field in HISTORY_FIELDS})


def write_training_plots(history: list[dict[str, float | int]], output_dir: Path) -> None:
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    _plot_lines(
        output_dir / "train_val_loss.png",
        epochs,
        {
            "train_loss": [float(row["train_loss"]) for row in history],
            "val_loss": [float(row["val_loss"]) for row in history],
        },
        ylabel="Loss",
    )
    _plot_lines(
        output_dir / "train_val_mean_iou.png",
        epochs,
        {"val_mean_iou": [float(row["val_mean_iou"]) for row in history]},
        ylabel="Mean IoU",
    )
    _plot_lines(
        output_dir / "learning_rate.png",
        epochs,
        {"learning_rate": [float(row["learning_rate"]) for row in history]},
        ylabel="Learning rate",
    )


def write_predictions_preview(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    output_path: Path,
    device: torch.device,
) -> None:
    try:
        images, masks = next(iter(dataloader))
    except StopIteration:
        return

    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
        predictions = logits.argmax(dim=1).cpu()

    image = images[0].permute(1, 2, 0).numpy()
    image = image.clip(0.0, 1.0)
    target = decode_train_ids_to_colors(masks[0].numpy().astype("uint8"))
    prediction = decode_train_ids_to_colors(predictions[0].numpy().astype("uint8"))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for axis, array, title in zip(
        axes,
        (image, target, prediction),
        ("Image", "Target", "Prediction"),
        strict=True,
    ):
        axis.imshow(array)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _progress_description(stage: str, epoch: int | None, total_epochs: int | None) -> str:
    if epoch is None or total_epochs is None:
        return stage
    return f"epoch {epoch}/{total_epochs} {stage}"


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


class GradScalerLike(Protocol):
    def scale(self, loss: torch.Tensor) -> Any:
        ...

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        ...

    def update(self) -> None:
        ...

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        ...


class NoOpGradScaler:
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        return None


def create_grad_scaler(use_amp: bool) -> GradScalerLike:
    if use_amp:
        try:
            return torch.amp.GradScaler("cuda")
        except AttributeError:
            return torch.cuda.amp.GradScaler()
    return NoOpGradScaler()


def autocast_context(use_amp: bool) -> Any:
    if use_amp:
        try:
            return torch.amp.autocast("cuda")
        except AttributeError:
            return torch.cuda.amp.autocast()
    return nullcontext()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir, history = train_model(config, run_name=args.run_name, resume=args.resume)
    last_epoch = history[-1]["epoch"] if history else "none"
    print(f"Training output directory: {output_dir}")
    print(f"Completed through epoch: {last_epoch}")


def _plot_lines(
    output_path: Path,
    epochs: list[int],
    series: dict[str, list[float]],
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in series.items():
        ax.plot(epochs, values, marker="o", label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _torch_load(path: Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


if __name__ == "__main__":
    main()
