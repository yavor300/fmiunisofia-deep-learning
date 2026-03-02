import ast
from pathlib import Path

import numpy as np


def load_literal_from_readme(var_name: str):
    readme_path = Path(__file__).with_name("README.md")
    content = readme_path.read_text(encoding="utf-8")
    assign_token = f"{var_name} ="
    assign_pos = content.find(assign_token)

    if assign_pos == -1:
        raise ValueError(f"Could not find `{var_name}` in {readme_path}")

    list_start = content.find("[", assign_pos)
    if list_start == -1:
        raise ValueError(f"Could not find list start for `{var_name}` in {readme_path}")

    depth = 0
    list_end = -1
    for index in range(list_start, len(content)):
        if content[index] == "[":
            depth += 1
        elif content[index] == "]":
            depth -= 1
            if depth == 0:
                list_end = index
                break

    if list_end == -1:
        raise ValueError(f"Could not find list end for `{var_name}` in {readme_path}")

    return ast.literal_eval(content[list_start:list_end + 1])


def print_summary_stats(column: np.ndarray, name: str):
    print(f"Summary statistics for {name}:")
    print(f"  count={column.size}")
    print(f"  min={column.min():.2f}")
    print(f"  max={column.max():.2f}")
    print(f"  mean={column.mean():.2f}")
    print(f"  median={np.median(column):.2f}")
    print(f"  std={column.std():.2f}")


def main():
    baseball_dataset = load_literal_from_readme("baseball_dataset")
    baseball = np.array(baseball_dataset)

    print(f"Number of rows and columns: {baseball.shape}")
    print_summary_stats(baseball[:, 0], "height")
    print_summary_stats(baseball[:, 1], "weight")
    print_summary_stats(baseball[:, 2], "age")
    print()

    corrected = baseball.copy()

    malformed_height_mask = corrected[:, 0] > 1000
    corrected[malformed_height_mask, 0] = corrected[malformed_height_mask, 0] / 1000

    print("After data correction:")
    print_summary_stats(corrected[:, 0], "height")
    print_summary_stats(corrected[:, 1], "weight")
    print_summary_stats(corrected[:, 2], "age")
    print()

    height = corrected[:, 0]
    weight = corrected[:, 1]
    height_weight_corr = np.corrcoef(height, weight)[0, 1]
    median_height = np.median(height)
    taller_mask = height > median_height
    mean_weight_taller = weight[taller_mask].mean()
    mean_weight_shorter = weight[~taller_mask].mean()

    print("Additional statistics:")
    print(f"Height/weight correlation: {height_weight_corr:.3f}")
    print(f"Median height: {median_height:.2f} inches")
    print(f"Mean weight above median height: {mean_weight_taller:.2f} lbs")
    print(f"Mean weight at or below median height: {mean_weight_shorter:.2f} lbs")


if __name__ == "__main__":
    main()
