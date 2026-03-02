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
    pop_2007_marker = "It lists the corresponding populations for the countries in 2007 in millions of people."
    colors_marker = "Using the list `colors` recreate the below plot."

    life_exp = load_list_from_readme("life_exp", econ_marker)
    gdp_cap = load_list_from_readme("gdp_cap", econ_marker)
    pop = np.array(load_list_from_readme("pop", pop_2007_marker))
    colors = load_list_from_readme("colors", colors_marker)

    china_idx = int(np.argmax(pop))
    india_idx = int(np.argsort(pop)[-2])

    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    plt.scatter(gdp_cap, life_exp, s=pop * 2, c=colors, alpha=0.8)
    plt.xscale("log")
    plt.xlabel("GDP per Capita [in USD]")
    plt.ylabel("Life Expectancy [in years]")
    plt.title("World Development in 2007")
    plt.text(gdp_cap[india_idx] * 0.82, life_exp[india_idx] + 3.0, "India")
    plt.text(gdp_cap[china_idx] * 0.82, life_exp[china_idx] + 3.0, "China")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("matplotlib-06-labeled-bubble-chart.png"), dpi=120)
    plt.close()

    # Correct choice: A.
    # Blue countries (Africa) cluster at lower GDP per capita and lower life expectancy.


if __name__ == "__main__":
    main()
