#!/usr/bin/env python3
import ast
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_list_from_readme(var_name: str, marker: str):
    readme_path = Path(__file__).with_name("README.md")
    content = readme_path.read_text(encoding="utf-8")

    marker_pos = content.find(marker)
    if marker_pos == -1:
        raise ValueError(f"Could not find marker `{marker}` in {readme_path}.")

    assign_pos = content.find(f"{var_name} =", marker_pos)
    if assign_pos == -1:
        raise ValueError(f"Could not find `{var_name}` after marker `{marker}`.")

    list_start = content.find("[", assign_pos)
    if list_start == -1:
        raise ValueError(f"Could not find list start for `{var_name}`.")

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
        raise ValueError(f"Could not find list end for `{var_name}`.")

    return ast.literal_eval(content[list_start:list_end + 1])


def main():
    econ_marker = "After that, analyze the following two lists:"
    life_1950_marker = "Then, let's compare the life expectancy in `2007` to the one observed in `1950`:"

    life_exp_2007 = load_list_from_readme("life_exp", econ_marker)
    life_exp_1950 = load_list_from_readme("life_exp1950", life_1950_marker)

    # Distribution analysis:
    # In 2007, most countries cluster roughly between 60 and 80 years,
    # with relatively fewer countries at very low life expectancy values.
    plt.figure(figsize=(9, 5))
    plt.hist(life_exp_2007, bins=15)
    plt.xlabel("Life Expectancy [in years]")
    plt.ylabel("Number of Countries")
    plt.title("Distribution of Life Expectancy in 2007")
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("matplotlib-03-life-expectancy-distribution.png"), dpi=120)
    plt.close()

    bins = np.linspace(20, 90, 16)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    axes[0].hist(life_exp_1950, bins=bins)
    axes[0].set_title("1950")
    axes[0].set_xlabel("Life Expectancy [in years]")
    axes[0].set_ylabel("Number of Countries")

    axes[1].hist(life_exp_2007, bins=bins)
    axes[1].set_title("2007")
    axes[1].set_xlabel("Life Expectancy [in years]")

    # Difference answer:
    # The 2007 distribution is shifted to the right higher life expectancy overall
    # and has far fewer countries in the very low life-expectancy range than 1950.
    fig.suptitle("Life Expectancy Comparison: 1950 vs 2007")
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("matplotlib-03-life-expectancy-1950-vs-2007.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
