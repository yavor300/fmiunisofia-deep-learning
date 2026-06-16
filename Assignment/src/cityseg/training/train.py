"""Training command entry point."""

from __future__ import annotations

import argparse

from src.cityseg.config import load_config, prepare_run


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("runtime", {})["stage"] = "train"
    output_dir, config_path = prepare_run(config, stage="train", run_name=args.run_name)
    print("Training will be implemented in a later phase.")
    print(f"Output directory: {output_dir}")
    print(f"Resolved config: {config_path}")


if __name__ == "__main__":
    main()
