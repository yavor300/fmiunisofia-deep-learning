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


def main():
    baseball = [180, 215, 210, 210, 188, 176, 209, 200]
    baseball_array = np.array(baseball)

    print("Sunday analysis:")
    print(f"Baseball array: {baseball_array}")
    print(f"Type of baseball array: {type(baseball_array)}")
    print()

    height_in = load_literal_from_readme("height_in")
    np_height_in = np.array(height_in)
    np_height_metres = np_height_in * 0.0254

    print("Monday analysis:")
    np_height_in_text = np.array2string(np_height_in, threshold=6, separator=", ")
    np_height_metres_text = np.array2string(np_height_metres, threshold=6, separator=", ")
    print(f"np_height_in=array({np_height_in_text}, shape={np_height_in.shape})")
    print(f"np_height_metres=array({np_height_metres_text}, shape={np_height_metres.shape})")


if __name__ == "__main__":
    main()
