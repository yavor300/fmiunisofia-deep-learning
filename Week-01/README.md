# Week-01

This repository contains introductory NumPy and Matplotlib exercises with simple neural-network-style models.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are using the local virtual environment in this project:

```bash
.venv/bin/pip install -r requirements.txt
```

## Project Structure

- `task01/empire_state_simulation.py`
  - Random-walk simulation (Empire State problem).
  - Output image: `task01/empire_state_simulation.png`

- `task02/multiply_by_two_nn.py`
  - NumPy-based neural model that learns `y = 2x`.
  - Output image: `task02/multiply_by_two_nn.png`

- `task03/logical_gates_nn.py`
  - Trains neural networks for logical gates (`AND`, `OR`, `NAND`, `XOR`).
  - Output images:
    - `task03/logical_gates_training_loss.png`
    - `task03/network_architecture.png`

- `task03_architecture/logical_gates_architecture.py`
  - Hand-crafted perceptron architecture for logic gates.
  - Implements XOR as `AND(OR(x1, x2), NAND(x1, x2))`.
  - Output image: `task03_architecture/xor_architecture.png`

- `task03_architecture/logical_gates_architecture_trainable.py`
  - Same 2-2-1 architecture, but with trainable weights (backpropagation) for XOR.
  - Output images:
    - `task03_architecture/xor_trainable_loss.png`
    - `task03_architecture/xor_trainable_architecture.png`

## Run Scripts

From the repository root:

```bash
.venv/bin/python task01/empire_state_simulation.py
.venv/bin/python task02/multiply_by_two_nn.py
.venv/bin/python task03/logical_gates_nn.py
.venv/bin/python task03_architecture/logical_gates_architecture.py
.venv/bin/python task03_architecture/logical_gates_architecture_trainable.py
```

If your environment provides `python` directly, you can replace `.venv/bin/python` with `python`.

## Notes

- Matplotlib is configured for file output (`Agg` backend), so plots are saved as image files.
- Console output includes metrics/truth tables for quick verification.
