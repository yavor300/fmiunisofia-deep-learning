from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from src.cityseg.reporting.build_model_report import build_model_report


class TestBuildModelReport(unittest.TestCase):
    def test_when_results_csv_exists_then_xlsx_preserves_first_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "experiment_results.csv"
            output_path = root / "model_report.xlsx"
            results_path.write_text(
                "experiment_id,model,split,mean_iou,mean_dice,pixel_accuracy,comments\n"
                "000_baseline_majority,majority_baseline,val,0.1,0.2,0.3,baseline\n"
                "001_other,unet,val,0.5,0.6,0.7,other\n",
                encoding="utf-8",
            )

            build_model_report(results_path, output_path)

            workbook = load_workbook(output_path)
            sheet = workbook.active
            self.assertEqual(sheet["A2"].value, "000_baseline_majority")
            self.assertEqual(sheet["A3"].value, "001_other")


if __name__ == "__main__":
    unittest.main()
