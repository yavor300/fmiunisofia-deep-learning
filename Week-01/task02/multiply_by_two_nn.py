import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

SEED = 42
N_SAMPLES = 512
LEARNING_RATE = 0.01
EPOCHS = 3000


def make_dataset(n_samples: int, rng: np.random.Generator):
    x = rng.uniform(-10.0, 10.0, size=(n_samples, 1))
    y = 2.0 * x
    return x, y


def train_linear_neuron(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    show_math: bool = False,
    math_epochs: int = 3,
    math_rows: int = 5,
):
    rng = np.random.default_rng(SEED)
    w = rng.normal(0.0, 0.1, size=(1, 1))
    b = np.zeros((1, 1))

    n = x.shape[0]
    losses = []

    for epoch in range(epochs):
        y_pred = x @ w + b
        error = y_pred - y
        loss = np.mean(error ** 2)
        losses.append(loss)

        dloss_dypred = (2.0 / n) * error
        dloss_dw = x.T @ dloss_dypred
        dloss_db = np.sum(dloss_dypred, axis=0, keepdims=True)

        if show_math and epoch < math_epochs:
            rows = min(math_rows, n)
            print(f"\n--- Matrix math (epoch {epoch}) ---")
            print(f"x shape={x.shape}, w shape={w.shape}, b shape={b.shape}, y shape={y.shape}")
            print("Forward: y_pred = x @ w + b")
            print(f"x[:{rows}] =\n{x[:rows]}")
            print(f"w =\n{w}")
            print(f"b =\n{b}")
            print(f"y_pred[:{rows}] =\n{y_pred[:rows]}")
            print(f"error[:{rows}] = y_pred - y =\n{error[:rows]}")
            print(f"loss = mean(error^2) = {loss:.10f}")

            print("\nBackward:")
            print("dloss_dypred = (2/n) * error")
            print(f"dloss_dypred[:{rows}] =\n{dloss_dypred[:rows]}")
            print("dloss_dw = x.T @ dloss_dypred")
            print(f"x.T shape={x.T.shape}, dloss_dypred shape={dloss_dypred.shape}")
            print(f"dloss_dw =\n{dloss_dw}")
            print("dloss_db = sum(dloss_dypred, axis=0, keepdims=True)")
            print(f"dloss_db =\n{dloss_db}")

        w -= learning_rate * dloss_dw
        b -= learning_rate * dloss_db

        if show_math and epoch < math_epochs:
            print("\nUpdate:")
            print("w = w - learning_rate * dloss_dw")
            print(f"new w =\n{w}")
            print("b = b - learning_rate * dloss_db")
            print(f"new b =\n{b}")

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | MSE: {loss:.10f}")

    return (w, b), np.array(losses)


def predict(x: np.ndarray, params):
    w, b = params
    return x @ w + b


def plot_results(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                 y_hat_test: np.ndarray, losses: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(losses, color="tab:blue")
    axes[0].set_title("Training Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(x_train[:, 0], y_train[:, 0], s=10, alpha=0.25, label="Train data")
    axes[1].plot(x_test[:, 0], y_test[:, 0], color="black", lw=2, label="True y = 2x")
    axes[1].plot(x_test[:, 0], y_hat_test[:, 0], color="tab:red", lw=2, ls="--", label="NN prediction")
    axes[1].set_title("Learned Function")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("task02/multiply_by_two_nn.png", dpi=150)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    x_train, y_train = make_dataset(N_SAMPLES, rng)

    params, losses = train_linear_neuron(
        x=x_train,
        y=y_train,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        show_math=True,
        math_epochs=2,
        math_rows=5,
    )

    x_test = np.linspace(-10.0, 10.0, 200).reshape(-1, 1)
    y_test = 2.0 * x_test
    y_hat_test = predict(x_test, params)

    w, b = params
    test_mse = np.mean((y_hat_test - y_test) ** 2)
    print(f"\nLearned parameters: w={w[0, 0]:.6f}, b={b[0, 0]:.6f}")
    print(f"Final test MSE: {test_mse:.12f}")

    sample_x = np.array([[-7.0], [-1.5], [0.0], [2.5], [9.0]])
    sample_y_hat = predict(sample_x, params)
    print("\nSample predictions:")
    for x_val, y_val in zip(sample_x[:, 0], sample_y_hat[:, 0]):
        print(f"x={x_val:5.2f} -> predicted={y_val:8.4f}, expected={2.0 * x_val:8.4f}")

    plot_results(x_train, y_train, x_test, y_test, y_hat_test, losses)


if __name__ == "__main__":
    main()
