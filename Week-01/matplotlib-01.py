#!/usr/bin/env python3
import ast
from pathlib import Path

import matplotlib

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
    world_marker = "The world bank has estimates of the world population"
    econ_marker = "After that, analyze the following two lists:"

    year = load_list_from_readme("year", world_marker)
    pop = load_list_from_readme("pop", world_marker)
    life_exp = load_list_from_readme("life_exp", econ_marker)
    gdp_cap = load_list_from_readme("gdp_cap", econ_marker)

    print(f"Last year with estimation: {year[-1]}")
    print(f"Last estimated population: {pop[-1]}")

    year_over_10_billion = next(y for y, population in zip(year, pop) if population > 10)
    _ = year_over_10_billion

    plt.figure(figsize=(9, 5))
    plt.plot(year, pop)
    plt.xlabel("Year")
    plt.ylabel("Population [in billions]")
    plt.title("World Population Projections")
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("matplotlib-01-population.png"), dpi=120)
    plt.close()

    # There is a clear positive relationship: countries with higher GDP per capita
    # generally have higher life expectancy, with diminishing gains at high GDP.
    plt.figure(figsize=(9, 5))
    plt.scatter(gdp_cap, life_exp)
    plt.xlabel("GDP per Capita [in USD]")
    plt.ylabel("Life Expectancy [in years]")
    plt.title("World Development in 2007")
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("matplotlib-01-world-development-2007.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
