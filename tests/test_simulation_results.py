import numpy as np
import pytest
from src.simulation_results import calculate_supply_meets_demand_probability


def test_equal_supply_and_demand():
    """
    As there is no reason for the supply or demand distribution to be greater than the other,
    the probability of supply meeting demand should be around 50%.
    """
    supply_samples = np.random.normal(500, 100, 5000)
    demand_samples = np.random.normal(500, 100, 5000)
    assert (
        49
        <= calculate_supply_meets_demand_probability(supply_samples, demand_samples)
        <= 51
    )


def test_supply_greater_than_demand():
    supply_samples = np.array([400, 500, 600])
    demand_samples = np.array([100, 200, 300])
    assert (
        calculate_supply_meets_demand_probability(supply_samples, demand_samples)
        == 100.0
    )


def test_supply_less_than_demand():
    supply_samples = np.array([100, 200, 300])
    demand_samples = np.array([400, 500, 600])
    assert (
        calculate_supply_meets_demand_probability(supply_samples, demand_samples) == 0.0
    )


def test_mixed_supply_and_demand():
    supply_samples = np.array([100, 500, 300])
    demand_samples = np.array([200, 200, 200])
    assert (
        calculate_supply_meets_demand_probability(supply_samples, demand_samples)
        == 66.66666666666666
    )


if __name__ == "__main__":
    pytest.main()
