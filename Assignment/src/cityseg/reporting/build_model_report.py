"""Build spreadsheet model reports from experiment result CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Font


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the model comparison report.")
    parser.add_argument(
        "--results",
        default="reports/experiment_results.csv",
        help="Experiment results CSV.",
    )
    parser.add_argument(
        "--output",
        default="reports/model_report.xlsx",
        help="Output XLSX report path.",
    )
    return parser.parse_args()


def build_model_report(
    results_path: str | Path = "reports/experiment_results.csv",
    output_path: str | Path = "reports/model_report.xlsx",
) -> Path:
    results = _read_results(Path(results_path))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Model Report"

    if not results:
        sheet.append(["experiment_id", "model", "split", "mean_iou", "mean_dice", "pixel_accuracy"])
    else:
        headers = list(results[0].keys())
        sheet.append(headers)
        for row in results:
            sheet.append([_coerce_cell(row.get(header, "")) for header in headers])

    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for column in sheet.columns:
        max_length = max(len(str(cell.value or "")) for cell in column)
        sheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 40)

    workbook.save(path)
    return path


def main() -> None:
    args = parse_args()
    output_path = build_model_report(results_path=args.results, output_path=args.output)
    print(f"Wrote model report: {output_path}")


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _coerce_cell(value: Any) -> Any:
    if value is None:
        return ""
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
