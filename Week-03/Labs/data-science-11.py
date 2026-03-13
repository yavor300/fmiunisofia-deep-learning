import numpy as np


np.random.seed(42)

learning_rates = np.random.uniform(0.0001, 0.01, size=10)
momentums = np.random.uniform(0.85, 0.99, size=10)

pairs = [(float(lr), float(momentum)) for lr, momentum in zip(learning_rates, momentums)]
print(pairs)
