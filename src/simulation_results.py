import numpy as np
import pandas as pd

# load data
supply_samples = pd.read_csv("./data/simulations/total_distributed.csv", header=None)
demand_samples = pd.read_csv("./data/simulations/vital_workers.csv", header=None)

# convert to numpy array
supply_samples = supply_samples.to_numpy()
demand_samples = demand_samples.to_numpy()

# Ensure both arrays have the same length
assert len(supply_samples) == len(
    demand_samples
), "Supply and demand samples must have the same length"

# Check overlap between distributions
print(f"Maximum value in supply distribution: {supply_samples.max()}")
print(f"Minimum value in demand distribution: {demand_samples.min()}")


def calculate_supply_meets_demand_probability(supply_samples, demand_samples):
    """
    Calculate probability that supply meets demand using direct sample comparison.

    Parameters:
    supply_samples: Array of supply quantities (5000 samples)
    demand_samples: Array of demand quantities (5000 samples)

    Returns:
    float: Probability that supply meets or exceeds demand
    """
    # Compare each supply sample with each demand sample
    # This creates a matrix of all possible supply-demand combinations
    supply_matrix = supply_samples[:, np.newaxis]  # shape: (5000, 1)
    sufficient = supply_matrix >= demand_samples  # shape: (5000, 5000)

    # Average over all combinations
    probability = np.mean(sufficient)

    return probability * 100


x = calculate_supply_meets_demand_probability(supply_samples, demand_samples)
print(f"Probability that supply meets demand: {x:.4f}%")
