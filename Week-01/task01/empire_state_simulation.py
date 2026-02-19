import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

N_WALKS = 100000
N_STEPS = 100
TARGET_STEP = 60
CLUMSY_PROB = 0.001
SEED = 25

def simulate_walks(n_walks: int, n_steps: int, target_step: int, seed: int = SEED):
    
    rng = np.random.default_rng(seed)
    all_positions = np.zeros((n_walks, n_steps + 1), dtype=int)
    for step in range(1, n_steps + 1):
        prev  = all_positions[:, step - 1].copy()
        die = rng.integers(1, 7, size=n_walks)
        down_mask = die <= 2
        up_one_mask = (die >= 3) & (die <= 5)
        up_jump_mask = die == 6
        prev[down_mask] = np.maximum(0, prev[down_mask] - 1)
        prev[up_one_mask] += 1
        prev[up_jump_mask] += rng.integers(1, 7, size=np.sum(up_jump_mask))
        clumsy_mask = rng.random(n_walks) <= CLUMSY_PROB
        prev[clumsy_mask] = 0
        all_positions[:, step] = prev

    final_positions = all_positions[:, -1]
    reached_target = np.any(all_positions >= target_step, axis=1)

    return final_positions, reached_target, all_positions

def main():
    final_positions, reached_target, all_positions = simulate_walks(
        n_walks=N_WALKS,
        n_steps=N_STEPS,
        target_step=TARGET_STEP,
        seed=SEED
    )

    prob_reach_ever = np.mean(reached_target)
    prob_end_at_or_above = np.mean(final_positions >= TARGET_STEP)

    print(f"Walks simulated: {N_WALKS:,}")
    print(f"Steps per walk: {N_STEPS}")
    print(f"Target step: {TARGET_STEP}")
    print(f"Clumsy fall probability: {CLUMSY_PROB}")
    print(f"P(reach step {TARGET_STEP} at least once) = {prob_reach_ever:.4%}")
    print(f"P(end at or above step {TARGET_STEP})   = {prob_end_at_or_above:.4%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sample_count = 15
    for i in range(sample_count):
        axes[0].plot(all_positions[i], lw=1)
    axes[0].axhline(TARGET_STEP, color="crimson", ls="--", lw=1.5, label=f"target={TARGET_STEP}")
    axes[0].set_title("Sample Random Walks")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Height")
    axes[0].legend()

    axes[1].hist(final_positions, bins=40, edgecolor="black", alpha=0.75)
    axes[1].axvline(TARGET_STEP, color="crimson", ls="--", lw=1.5, label=f"target={TARGET_STEP}")
    axes[1].set_title("Final Position Distribution")
    axes[1].set_xlabel(f"Final height after {N_STEPS} steps")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("task01/empire_state_simulation.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
