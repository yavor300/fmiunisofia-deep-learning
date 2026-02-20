import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

SEED = 7
LEARNING_RATE = 0.8
EPOCHS = 12000
HIDDEN_SIZE = 4


X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
)

GATES = {
    "AND": np.array([[0.0], [0.0], [0.0], [1.0]]),
    "OR": np.array([[0.0], [1.0], [1.0], [1.0]]),
    "NAND": np.array([[1.0], [1.0], [1.0], [0.0]]),
    "XOR": np.array([[0.0], [1.0], [1.0], [0.0]]),
}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class TwoLayerNN:
    def __init__(self, input_size: int, hidden_size: int, seed: int):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.8, size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = rng.normal(0.0, 0.8, size=(hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, x: np.ndarray):
        z1 = x @ self.w1 + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = sigmoid(z2)
        return z1, a1, z2, a2

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float):
        m = x.shape[0]
        losses = []
        eps = 1e-9

        for _ in range(epochs):
            _, a1, _, y_hat = self.forward(x)

            # Binary cross-entropy
            y_hat_clip = np.clip(y_hat, eps, 1.0 - eps)
            loss = -np.mean(y * np.log(y_hat_clip) + (1.0 - y) * np.log(1.0 - y_hat_clip))
            losses.append(loss)

            # Backprop for sigmoid + BCE output
            dz2 = y_hat - y
            dw2 = (a1.T @ dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m

            da1 = dz2 @ self.w2.T
            dz1 = da1 * a1 * (1.0 - a1)
            dw1 = (x.T @ dz1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m

            self.w2 -= lr * dw2
            self.b2 -= lr * db2
            self.w1 -= lr * dw1
            self.b1 -= lr * db1

        return np.array(losses)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[-1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


def print_truth_table(gate_name: str, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    print(f"\n{gate_name} truth table")
    print("A B | true pred prob")
    for i in range(x.shape[0]):
        a, b = x[i].astype(int)
        t = int(y_true[i, 0])
        p = int(y_pred[i, 0])
        pr = float(y_prob[i, 0])
        print(f"{a} {b} |   {t}    {p}   {pr:.4f}")


def plot_losses(losses_by_gate):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (gate, losses) in zip(axes.flatten(), losses_by_gate.items()):
        ax.plot(losses, lw=2)
        ax.set_title(f"{gate} training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE loss")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("task03/logical_gates_training_loss.png", dpi=150)
    plt.close(fig)


def plot_network_architecture(hidden_size: int):
    fig, ax = plt.subplots(figsize=(10, 6))

    x_in, x_hidden, x_out = 0.15, 0.5, 0.85
    input_y = np.array([0.35, 0.65])
    hidden_y = np.linspace(0.2, 0.8, hidden_size)
    output_y = np.array([0.5])

    for yi in input_y:
        for yh in hidden_y:
            ax.plot([x_in, x_hidden], [yi, yh], color="gray", alpha=0.45, lw=1.0)
    for yh in hidden_y:
        ax.plot([x_hidden, x_out], [yh, output_y[0]], color="gray", alpha=0.45, lw=1.0)

    ax.scatter([x_in] * len(input_y), input_y, s=700, color="#4C72B0", edgecolors="black", zorder=3)
    ax.scatter([x_hidden] * len(hidden_y), hidden_y, s=700, color="#55A868", edgecolors="black", zorder=3)
    ax.scatter([x_out], output_y, s=700, color="#C44E52", edgecolors="black", zorder=3)

    input_labels = ["A", "B"]
    for yi, label in zip(input_y, input_labels):
        ax.text(x_in, yi, label, ha="center", va="center", fontsize=12, color="white", weight="bold")
    for idx, yh in enumerate(hidden_y, start=1):
        ax.text(x_hidden, yh, f"H{idx}", ha="center", va="center", fontsize=11, color="white", weight="bold")
    ax.text(x_out, output_y[0], "Y", ha="center", va="center", fontsize=12, color="white", weight="bold")

    ax.text(x_in, 0.9, "Input layer", ha="center", fontsize=12, weight="bold")
    ax.text(x_hidden, 0.9, f"Hidden layer ({hidden_size})", ha="center", fontsize=12, weight="bold")
    ax.text(x_out, 0.9, "Output layer", ha="center", fontsize=12, weight="bold")

    ax.set_title("Neural Network Architecture for Logic Gates (2 -> hidden -> 1)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.05, 0.98)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("task03/network_architecture.png", dpi=150)
    plt.close(fig)


def main():
    losses_by_gate = {}

    for idx, (gate_name, y) in enumerate(GATES.items()):
        model = TwoLayerNN(input_size=2, hidden_size=HIDDEN_SIZE, seed=SEED + idx)
        losses = model.train(X, y, epochs=EPOCHS, lr=LEARNING_RATE)

        y_prob = model.predict_proba(X)
        y_pred = model.predict(X)
        accuracy = np.mean(y_pred == y)

        print(f"{gate_name} accuracy: {accuracy:.2%}")
        print_truth_table(gate_name, X, y, y_pred, y_prob)

        losses_by_gate[gate_name] = losses

    plot_losses(losses_by_gate)
    plot_network_architecture(HIDDEN_SIZE)
    print("\nSaved plot: task03/logical_gates_training_loss.png")
    print("Saved plot: task03/network_architecture.png")


if __name__ == "__main__":
    main()
