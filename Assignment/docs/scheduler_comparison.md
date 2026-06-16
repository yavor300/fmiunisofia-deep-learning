# Scheduler Comparison

Phase 9 adds two matched Tiny U-Net experiments for comparing learning-rate schedules:

| Config | Scheduler | Purpose |
| --- | --- | --- |
| `configs/experiments/001_tiny_unet_ce_step.yaml` | Step decay | Drops LR by `gamma` every `step_size` epochs. |
| `configs/experiments/002_tiny_unet_ce_cosine.yaml` | Cosine annealing | Smoothly anneals LR toward `eta_min`. |

The training history CSV records `learning_rate` for every epoch. Each run also writes:

- `learning_rate.png`
- `train_val_loss.png`
- `train_val_mean_iou.png`

Use those curves to compare whether abrupt step drops or smooth cosine decay give better validation mean IoU for the same model and loss.
