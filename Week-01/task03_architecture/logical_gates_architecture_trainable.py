import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
)

Y_XOR = np.array([[0.0], [1.0], [1.0], [0.0]])

LEARNING_RATE = 1.0
EPOCHS = 20000
MAX_RESTARTS = 50


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


class TrainableArchitectureXOR:
    """
    Same topology as the hand-crafted diagram:
      x1,x2 -> [h1, h2] -> y
    Here h1/h2 are trainable neurons (can learn OR-like / NAND-like roles).
    """

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.5, size=(2, 2))  # [[w11,w13],[w12,w14]] by columns
        self.b1 = np.zeros((1, 2))
        self.w2 = rng.normal(0.0, 0.5, size=(2, 1))  # [[w21],[w22]]
        self.b2 = np.zeros((1, 1))                    # b3

    def forward(self, x: np.ndarray):
        z1 = x @ self.w1 + self.b1
        h = sigmoid(z1)
        z2 = h @ self.w2 + self.b2
        y_hat = sigmoid(z2)
        return z1, h, z2, y_hat

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float):
        m = x.shape[0]
        losses = []

        for _ in range(epochs):
            _, h, _, y_hat = self.forward(x)
            losses.append(bce_loss(y, y_hat))

            # BCE + sigmoid output derivative simplifies to (y_hat - y)
            dz2 = y_hat - y
            dw2 = (h.T @ dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m

            dh = dz2 @ self.w2.T
            dz1 = dh * h * (1.0 - h)
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


def try_train_until_success(x: np.ndarray, y: np.ndarray):
    for seed in range(MAX_RESTARTS):
        model = TrainableArchitectureXOR(seed=seed)
        losses = model.train(x, y, epochs=EPOCHS, lr=LEARNING_RATE)
        y_pred = model.predict(x)
        acc = float(np.mean(y_pred == y))
        if acc == 1.0:
            return model, losses, seed
    return None, None, None


def print_truth_table(x: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray):
    print("\nXOR truth table (trained model)")
    print("x1 x2 | true pred prob")
    for i in range(x.shape[0]):
        x1, x2 = x[i].astype(int)
        t = int(y_true[i, 0])
        p = int(y_pred[i, 0])
        pr = float(y_prob[i, 0])
        print(f" {x1}  {x2} |   {t}    {p}   {pr:.4f}")


def print_learned_parameters(model: TrainableArchitectureXOR):
    # Mapping to your drawing labels
    # h1 receives w11 (from x1), w12 (from x2), bias b1
    # h2 receives w13 (from x1), w14 (from x2), bias b2
    # y receives w21 (from h1), w22 (from h2), bias b3
    w11 = model.w1[0, 0]
    w12 = model.w1[1, 0]
    w13 = model.w1[0, 1]
    w14 = model.w1[1, 1]
    b1 = model.b1[0, 0]
    b2 = model.b1[0, 1]
    w21 = model.w2[0, 0]
    w22 = model.w2[1, 0]
    b3 = model.b2[0, 0]

    print("\nLearned parameters (mapped to diagram labels):")
    print(f"w11={w11:.4f}, w12={w12:.4f}, b1={b1:.4f}")
    print(f"w13={w13:.4f}, w14={w14:.4f}, b2={b2:.4f}")
    print(f"w21={w21:.4f}, w22={w22:.4f}, b3={b3:.4f}")


def plot_training_loss(losses: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, lw=2, color="tab:blue")
    ax.set_title("Trainable XOR Architecture - BCE Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("task03_architecture/xor_trainable_loss.png", dpi=150)
    plt.close(fig)


def plot_learned_architecture(model: TrainableArchitectureXOR):
    fig, ax = plt.subplots(figsize=(12, 6))

    x_input = 0.10
    x_hidden = 0.45
    x_output = 0.78
    x_final = 0.92

    y_x1, y_x2 = 0.70, 0.30
    y_h1, y_h2, y_out = 0.72, 0.28, 0.50

    ax.text(x_input - 0.05, y_x1, "x1", fontsize=16, weight="bold", va="center")
    ax.text(x_input - 0.05, y_x2, "x2", fontsize=16, weight="bold", va="center")

    h1_circle = plt.Circle((x_hidden, y_h1), 0.11, fill=False, lw=2, color="#1f77b4")
    h2_circle = plt.Circle((x_hidden, y_h2), 0.11, fill=False, lw=2, color="#1f77b4")
    out_circle = plt.Circle((x_output, y_out), 0.11, fill=False, lw=2, color="#1f77b4")
    ax.add_patch(h1_circle)
    ax.add_patch(h2_circle)
    ax.add_patch(out_circle)

    ax.text(x_hidden, y_h1, "H1", ha="center", va="center", fontsize=16, color="#1f77b4", weight="bold")
    ax.text(x_hidden, y_h2, "H2", ha="center", va="center", fontsize=16, color="#1f77b4", weight="bold")
    ax.text(x_output, y_out, "Y", ha="center", va="center", fontsize=16, color="#1f77b4", weight="bold")

    arrow = dict(arrowstyle="->", lw=2, color="#1f77b4")
    ax.annotate("", xy=(x_hidden - 0.11, y_h1), xytext=(x_input, y_x1), arrowprops=arrow)
    ax.annotate("", xy=(x_hidden - 0.11, y_h1 - 0.02), xytext=(x_input, y_x2), arrowprops=arrow)
    ax.annotate("", xy=(x_hidden - 0.11, y_h2 + 0.02), xytext=(x_input, y_x1), arrowprops=arrow)
    ax.annotate("", xy=(x_hidden - 0.11, y_h2), xytext=(x_input, y_x2), arrowprops=arrow)
    ax.annotate("", xy=(x_output - 0.11, y_out + 0.01), xytext=(x_hidden + 0.11, y_h1), arrowprops=arrow)
    ax.annotate("", xy=(x_output - 0.11, y_out - 0.01), xytext=(x_hidden + 0.11, y_h2), arrowprops=arrow)
    ax.annotate("", xy=(x_final, y_out), xytext=(x_output + 0.11, y_out), arrowprops=arrow)

    ax.annotate("b1", xy=(x_hidden - 0.08, y_h1 + 0.09), xytext=(x_hidden - 0.16, y_h1 + 0.16),
                fontsize=13, arrowprops=dict(arrowstyle="->", lw=1.2, ls="--", color="gray"))
    ax.annotate("b2", xy=(x_hidden - 0.08, y_h2 - 0.09), xytext=(x_hidden - 0.16, y_h2 - 0.16),
                fontsize=13, arrowprops=dict(arrowstyle="->", lw=1.2, ls="--", color="gray"))
    ax.annotate("b3", xy=(x_output - 0.08, y_out + 0.09), xytext=(x_output - 0.13, y_out + 0.16),
                fontsize=13, arrowprops=dict(arrowstyle="->", lw=1.2, ls="--", color="gray"))

    # Label connections with learned values
    ax.text(0.21, 0.77, f"w11={model.w1[0,0]:.2f}", fontsize=11, color="#1f77b4")
    ax.text(0.21, 0.56, f"w12={model.w1[1,0]:.2f}", fontsize=11, color="#1f77b4")
    ax.text(0.21, 0.46, f"w13={model.w1[0,1]:.2f}", fontsize=11, color="#1f77b4")
    ax.text(0.21, 0.24, f"w14={model.w1[1,1]:.2f}", fontsize=11, color="#1f77b4")
    ax.text(0.57, 0.63, f"w21={model.w2[0,0]:.2f}", fontsize=11, color="#1f77b4")
    ax.text(0.57, 0.37, f"w22={model.w2[1,0]:.2f}", fontsize=11, color="#1f77b4")

    ax.text(x_final + 0.01, y_out, "y", fontsize=16, weight="bold", va="center")
    ax.set_title("Trained XOR Network (same 2-2-1 architecture)", fontsize=16)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.05, 0.95)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("task03_architecture/xor_trainable_architecture.png", dpi=150)
    plt.close(fig)


def main():
    model, losses, seed = try_train_until_success(X, Y_XOR)
    if model is None:
        raise RuntimeError("Could not reach 100% XOR accuracy; increase EPOCHS or MAX_RESTARTS.")

    y_prob = model.predict_proba(X)
    y_pred = model.predict(X)
    acc = np.mean(y_pred == Y_XOR)

    print(f"Found successful training run with seed={seed}")
    print(f"Final accuracy: {acc:.2%}")
    print(f"Final loss: {losses[-1]:.8f}")

    print_truth_table(X, Y_XOR, y_prob, y_pred)
    print_learned_parameters(model)

    plot_training_loss(losses)
    plot_learned_architecture(model)
    print("\nSaved: task03_architecture/xor_trainable_loss.png")
    print("Saved: task03_architecture/xor_trainable_architecture.png")


if __name__ == "__main__":
    main()
