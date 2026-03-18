"""
PPE (P4E) model: vital worker requirements and rapidly-accessible PPE availability.

Replicates the Guesstimate model "Rapidly-accessible P4E availability and requirements".
Monte Carlo runs produce supply (total distributed P4E) and demand (units P4E needed)
so P(supply >= demand) can be computed.

All user-editable parameters are in PARAMETERS below; formulas follow the Guesstimate logic.
"""

import csv
import os
import numpy as np

try:
    from .mc_distributions import sample_normal, sample_lognormal, sample_uniform
except ImportError:
    from mc_distributions import sample_normal, sample_lognormal, sample_uniform


# ---------------------------------------------------------------------------
# PARAMETERS (edit these to change assumptions; ranges are [low, high] for distributions)
# ---------------------------------------------------------------------------
PARAMETERS = {
    # Section 1: People required
    "usps_non_mail_carriers_total": {"dist": "point", "value": 173_240},
    "usps_mail_carriers_total": {"dist": "point", "value": 326_760},
    "proportion_needed_non_mail": {"dist": "lognormal", "low": 0.05, "high": 0.2},
    "proportion_needed_mail": {"dist": "lognormal", "low": 0.05, "high": 0.2},
    "telecomms_total": {"dist": "point", "value": 250_000},
    "proportion_needed_telecomms": {"dist": "lognormal", "low": 0.05, "high": 0.2},
    "ppe_workers_total": {"dist": "point", "value": 16_819},
    "proportion_needed_ppe_workers": {"dist": "lognormal", "low": 1.0, "high": 1.5},
    "law_enforcement_total": {"dist": "lognormal", "low": 450_000, "high": 660_000},
    "energy_workers_total": {"dist": "point", "value": 3_283_300},
    "proportion_needed_energy_workers": {
        "dist": "lognormal",
        "low": 0.01,
        "high": 0.10,
    },
    "us_total_energy_consumption": {"dist": "point", "value": 4.18e12},
    # Section 2: National PPE
    "military_m50_stockpile": {"dist": "lognormal", "low": 156_170, "high": 1_700_000},
    "sns_ehmrs": {"dist": "normal", "low": 139_500, "high": 558_000},
    "sns_n95s": {"dist": "point", "value": 538_000_000},
    "papr_spending": {"dist": "point", "value": 274_306_387},
    "papr_unit_cost": {"dist": "normal", "low": 1000, "high": 2000},
    "ratio_purchase_to_stockpile": {"dist": "normal", "low": 0.5, "high": 2},
    "proportion_national_paprs": {"dist": "normal", "low": 0.25, "high": 0.5},
    # Section 3: Local
    "paprs_stock_on_hand": {"dist": "normal", "low": 65_770, "high": 312_886},
    "proportion_facilities_with_elastomerics": {
        "dist": "normal",
        "low": 0.06,
        "high": 0.12,
    },
    "elastomerics_needed_flu_pandemic": {
        "dist": "normal",
        "low": 999_000,
        "high": 3_533_796,
    },
    "actual_facilities_stocked": {"dist": "normal", "low": 0.5, "high": 1.0},
    # Section 4: Inventory
    "us_papr_market": {"dist": "normal", "low": 110_880_000, "high": 770_000_000},
    "papr_cost_inventory": {"dist": "normal", "low": 1000, "high": 2000},
    "suitability": {"dist": "normal", "low": 0.70, "high": 1.0},
    "days_on_hand": {"dist": "normal", "low": 30, "high": 80},
    "n95_sales_pre_covid": {"dist": "normal", "low": 400_000_000, "high": 600_000_000},
    "elastomeric_sales_vs_n95": {"dist": "normal", "low": 0.33, "high": 0.67},
    "elastomeric_unit_cost": {"dist": "normal", "low": 25, "high": 50},
    "proportion_suitable_filters": {"dist": "lognormal", "low": 0.50, "high": 0.75},
    # Section 5: Total distributed (private sector)
    "papr_private_sector_employees": {"dist": "point", "value": 287_333},
    "paprs_per_papr_employee": {"dist": "lognormal", "low": 0.25, "high": 1.0},
    "non_powered_private_sector_employees": {"dist": "point", "value": 3_067_325},
    "ehmr_usage_orgs": {"dist": "point", "value": 0.486},
    "employee_org_usage_ratio": {"dist": "normal", "low": 0.58, "high": 0.99},
    "elastomerics_per_elastomeric_employee": {"dist": "lognormal", "low": 1, "high": 4},
    "efmr_usage_orgs": {"dist": "point", "value": 0.218},
    # Section 6: Vital workers (demand)
    "all_essential_workers": {"dist": "point", "value": 55_000_000},
    "proportion_vital_vs_essential": {"dist": "lognormal", "low": 0.25, "high": 0.82},
    "proportion_needing_p4e": {"dist": "normal", "low": 0.70, "high": 1.0},
    "units_p4e_per_worker": {"dist": "lognormal", "low": 1, "high": 2},
}

DEFAULT_N_SAMPLES = 5000
CONFIDENCE = 90
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _sample_param(name: str, n: int, params: dict) -> np.ndarray:
    """Sample a single parameter; returns array of shape (n,)."""
    p = params.get(name, PARAMETERS[name]).copy()
    dist = p.get("dist", "point")
    if dist == "point":
        return np.full(n, p["value"])
    low, high = p["low"], p["high"]
    if dist == "normal":
        return sample_normal(low, high, n, confidence=CONFIDENCE)
    if dist == "lognormal":
        return sample_lognormal(low, high, n, confidence=CONFIDENCE)
    if dist == "uniform":
        return sample_uniform(low, high, n)
    raise ValueError(f"Unknown dist '{dist}' for {name}")


def _sample_all_params(n: int, user_params: dict) -> dict:
    """Sample all root parameters; returns dict of name -> array (length n)."""
    n = int(n)
    out = {}
    for key in PARAMETERS:
        out[key] = _sample_param(key, n, user_params)
    return out


def compute_stockpiled_ppe(sampled: dict) -> dict:
    """
    Stockpiled PPE available, split by military, national (excluding military), and local.

    Returns dict with keys: military, national_excl_military, local, total.
    """
    n = len(sampled["military_m50_stockpile"])
    military_m50_stockpile = sampled["military_m50_stockpile"]
    sns_ehmrs = sampled["sns_ehmrs"]
    papr_spending = sampled["papr_spending"]
    papr_unit_cost = sampled["papr_unit_cost"]
    ratio_purchase_to_stockpile = sampled["ratio_purchase_to_stockpile"]
    proportion_national_paprs = sampled["proportion_national_paprs"]
    paprs_purchased = papr_spending / papr_unit_cost
    paprs_local_and_national = paprs_purchased * ratio_purchase_to_stockpile
    sns_paprs = paprs_local_and_national * proportion_national_paprs

    military = military_m50_stockpile
    national_excl_military = sns_ehmrs + sns_paprs

    paprs_stock_on_hand = sampled["paprs_stock_on_hand"]
    proportion_facilities_with_elastomerics = sampled[
        "proportion_facilities_with_elastomerics"
    ]
    elastomerics_needed_flu_pandemic = sampled["elastomerics_needed_flu_pandemic"]
    actual_facilities_stocked = sampled["actual_facilities_stocked"]
    total_elastomerics_stocked_if_flu = (
        proportion_facilities_with_elastomerics * elastomerics_needed_flu_pandemic
    )
    units_stockpiled_national_elastomerics = (
        total_elastomerics_stocked_if_flu * actual_facilities_stocked
    )
    local_paprs = paprs_stock_on_hand * np.ones(n)
    local_elastomerics = units_stockpiled_national_elastomerics
    local = local_paprs + local_elastomerics

    total = military + national_excl_military + local
    return {
        "military": military,
        "national_excl_military": national_excl_military,
        "local": local,
        "total": total,
    }


def compute_inventory_ppe(sampled: dict) -> np.ndarray:
    """Inventory PPE available (PAPRs + elastomerics on hand in supply chain)."""
    us_papr_market = sampled["us_papr_market"]
    papr_cost_inventory = sampled["papr_cost_inventory"]
    suitability = sampled["suitability"]
    days_on_hand = sampled["days_on_hand"]
    annual_papr_product = us_papr_market / papr_cost_inventory
    annual_suitable_papr_product = annual_papr_product * suitability
    paprs_inventory_on_hand = annual_suitable_papr_product * (days_on_hand / 365)

    n95_sales_pre_covid = sampled["n95_sales_pre_covid"]
    elastomeric_sales_vs_n95 = sampled["elastomeric_sales_vs_n95"]
    elastomeric_unit_cost = sampled["elastomeric_unit_cost"]
    proportion_suitable_filters = sampled["proportion_suitable_filters"]
    elastomeric_sales = n95_sales_pre_covid * elastomeric_sales_vs_n95
    annual_elastomerics = elastomeric_sales / elastomeric_unit_cost
    annual_suitable_elastomerics = annual_elastomerics * proportion_suitable_filters
    elastomerics_inventory_on_hand = annual_suitable_elastomerics * (days_on_hand / 365)

    return paprs_inventory_on_hand + elastomerics_inventory_on_hand


def compute_total_distributed(
    sampled: dict, stockpiled: dict, inventory_ppe: np.ndarray
) -> dict:
    """
    Total distributed PPE: private sector contribution and overall supply.

    Returns dict with keys: private_sector, total_supply.
    """
    n = len(inventory_ppe)
    total_stockpiled = stockpiled["total"]

    papr_private_sector_employees = sampled["papr_private_sector_employees"]
    paprs_per_papr_employee = sampled["paprs_per_papr_employee"]
    total_private_sector_paprs = papr_private_sector_employees * paprs_per_papr_employee

    non_powered_private_sector_employees = sampled[
        "non_powered_private_sector_employees"
    ]
    ehmr_usage_orgs = sampled["ehmr_usage_orgs"]
    employee_org_usage_ratio = sampled["employee_org_usage_ratio"]
    elastomerics_per_elastomeric_employee = sampled[
        "elastomerics_per_elastomeric_employee"
    ]
    ehmr_usage_employees = (
        non_powered_private_sector_employees
        * ehmr_usage_orgs
        * employee_org_usage_ratio
    )
    total_private_sector_ehmrs = (
        ehmr_usage_employees * elastomerics_per_elastomeric_employee
    )
    efmr_usage_orgs = sampled["efmr_usage_orgs"]
    efmr_usage_employees = (
        non_powered_private_sector_employees
        * efmr_usage_orgs
        * employee_org_usage_ratio
    )
    total_private_sector_efmrs = (
        efmr_usage_employees * elastomerics_per_elastomeric_employee
    )
    total_private_sector_elastomerics = (
        total_private_sector_ehmrs + total_private_sector_efmrs
    )

    private_sector = total_private_sector_paprs + total_private_sector_elastomerics
    total_supply = total_stockpiled + inventory_ppe + private_sector
    return {"private_sector": private_sector, "total_supply": total_supply}


def compute_vital_workers_demand(sampled: dict) -> dict:
    """
    Total vital workers and their PPE demand.

    Returns dict with keys: vital_workers_count, vital_workers_ppe_demand.
    """
    all_essential_workers = sampled["all_essential_workers"]
    proportion_vital_vs_essential = sampled["proportion_vital_vs_essential"]
    proportion_needing_p4e = sampled["proportion_needing_p4e"]
    units_p4e_per_worker = sampled["units_p4e_per_worker"]
    all_vital_workers = all_essential_workers * proportion_vital_vs_essential
    vital_workers_needing_p4e = all_vital_workers * proportion_needing_p4e
    vital_workers_ppe_demand = vital_workers_needing_p4e * units_p4e_per_worker
    return {
        "vital_workers_count": all_vital_workers,
        "vital_workers_ppe_demand": vital_workers_ppe_demand,
    }


def compute_people_required_demand(sampled: dict) -> dict:
    """
    People required to produce more PPE (sector workers) and their PPE demand.

    Returns dict with keys: people_required_worker_count, people_required_ppe_demand.
    """
    n = len(sampled["usps_non_mail_carriers_total"])
    usps_non_mail_carriers_total = sampled["usps_non_mail_carriers_total"]
    usps_mail_carriers_total = sampled["usps_mail_carriers_total"]
    proportion_needed_non_mail = sampled["proportion_needed_non_mail"]
    proportion_needed_mail = sampled["proportion_needed_mail"]
    usps_non_mail_needed = usps_non_mail_carriers_total * proportion_needed_non_mail
    usps_mail_needed = usps_mail_carriers_total * proportion_needed_mail
    ffr_usps_non_mail = np.zeros(n)
    ffr_usps_mail = usps_mail_needed
    ehmr_usps_non_mail = usps_non_mail_needed
    ehmr_usps_mail = np.zeros(n)

    telecomms_total = sampled["telecomms_total"]
    proportion_needed_telecomms = sampled["proportion_needed_telecomms"]
    telecomms_needed = telecomms_total * proportion_needed_telecomms
    ffr_telecomms = 0.5 * telecomms_needed
    ehmr_telecomms = 0.5 * telecomms_needed

    ppe_workers_total = sampled["ppe_workers_total"]
    proportion_needed_ppe_workers = sampled["proportion_needed_ppe_workers"]
    ppe_workers_needed = ppe_workers_total * proportion_needed_ppe_workers
    ffr_ppe_workers = np.zeros(n)
    ehmr_ppe_workers = ppe_workers_needed

    law_enforcement_total = sampled["law_enforcement_total"]
    ffr_law_enforcement = 0.3 * law_enforcement_total
    ehmr_law_enforcement = 0.7 * law_enforcement_total

    energy_workers_total = sampled["energy_workers_total"]
    proportion_needed_energy_workers = sampled["proportion_needed_energy_workers"]
    energy_workers_needed = energy_workers_total * proportion_needed_energy_workers
    ehmr_energy_workers = energy_workers_needed

    group_ffrs_total = (
        ffr_usps_non_mail
        + ffr_usps_mail
        + ffr_telecomms
        + ffr_ppe_workers
        + ffr_law_enforcement
    )
    group_ehmrs_total = (
        ehmr_usps_non_mail
        + ehmr_usps_mail
        + ehmr_telecomms
        + ehmr_ppe_workers
        + ehmr_law_enforcement
        + ehmr_energy_workers
    )
    people_required_worker_count = group_ffrs_total + group_ehmrs_total
    units_p4e_per_worker = sampled["units_p4e_per_worker"]
    people_required_ppe_demand = group_ehmrs_total * units_p4e_per_worker
    return {
        "people_required_worker_count": people_required_worker_count,
        "people_required_ppe_demand": people_required_ppe_demand,
    }


def _median_90ci(arr: np.ndarray) -> tuple[float, float, float]:
    """Return (median, 5th percentile, 95th percentile) for 90% CI."""
    return (
        float(np.median(arr)),
        float(np.percentile(arr, 5)),
        float(np.percentile(arr, 95)),
    )


def run_ppe_mc(
    n: int = DEFAULT_N_SAMPLES, params: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Monte Carlo: sample all parameters and compute supply and demand.

    Returns:
        demand_samples: Units P4E needed for vital workers (length n).
        supply_samples: Total rapidly-accessible P4E (length n).
    """
    user_params = params if params is not None else {}
    n = int(n)
    sampled = _sample_all_params(n, user_params)

    stockpiled = compute_stockpiled_ppe(sampled)
    inventory_ppe = compute_inventory_ppe(sampled)
    distributed = compute_total_distributed(sampled, stockpiled, inventory_ppe)
    vital = compute_vital_workers_demand(sampled)
    people_req = compute_people_required_demand(sampled)

    supply_samples = distributed["total_supply"]
    demand_samples = vital["vital_workers_ppe_demand"]
    return demand_samples, supply_samples


def _compute_medians_for_params(
    n: int, params: dict | None = None
) -> tuple[float, float]:
    """
    Helper for sensitivity analysis: run Monte Carlo with the given parameter
    overrides and return median units needed and median total supply.
    """
    user_params = params if params is not None else {}
    n = int(n)
    sampled = _sample_all_params(n, user_params)

    stockpiled = compute_stockpiled_ppe(sampled)
    inventory_ppe = compute_inventory_ppe(sampled)
    distributed = compute_total_distributed(sampled, stockpiled, inventory_ppe)
    vital = compute_vital_workers_demand(sampled)

    units_needed_median = float(np.median(vital["vital_workers_ppe_demand"]))
    total_supply_median = float(np.median(distributed["total_supply"]))
    return units_needed_median, total_supply_median


def run_one_way_sensitivity(n: int = DEFAULT_N_SAMPLES, seed: int | None = 42) -> dict:
    """
    One-way (low/high) sensitivity analysis on root PARAMETERS.

    For each parameter with low/high bounds, fixes that parameter at its low
    and high values (point distributions) while leaving all others as in
    PARAMETERS, and computes the change in median units needed and total
    rapidly-accessible P4E supply relative to the baseline.

    Returns a dict with structure:
        {
            "units_needed": {
                name: {"baseline": ..., "delta_low": ..., "delta_high": ...},
                ...
            },
            "total_supply": {
                name: {"baseline": ..., "delta_low": ..., "delta_high": ...},
                ...
            },
        }
    """
    if seed is not None:
        np.random.seed(seed)

    baseline_units, baseline_supply = _compute_medians_for_params(n, params=None)

    results_units: dict[str, dict[str, float]] = {}
    results_supply: dict[str, dict[str, float]] = {}

    # Parameters that directly affect vital-workers demand
    demand_params = {
        "all_essential_workers",
        "proportion_vital_vs_essential",
        "proportion_needing_p4e",
        "units_p4e_per_worker",
    }

    for name, spec in PARAMETERS.items():
        dist = spec.get("dist", "point")
        has_bounds = "low" in spec and "high" in spec

        # Skip parameters without low/high bounds; they would have zero bar length.
        if dist == "point" or not has_bounds:
            continue

        low_val = spec["low"]
        high_val = spec["high"]

        # Low scenario
        params_low = {name: {"dist": "point", "value": low_val}}
        units_low, supply_low = _compute_medians_for_params(n, params=params_low)

        # High scenario
        params_high = {name: {"dist": "point", "value": high_val}}
        units_high, supply_high = _compute_medians_for_params(n, params=params_high)

        if name in demand_params:
            results_units[name] = {
                "baseline": baseline_units,
                "delta_low": units_low - baseline_units,
                "delta_high": units_high - baseline_units,
            }

        results_supply[name] = {
            "baseline": baseline_supply,
            "delta_low": supply_low - baseline_supply,
            "delta_high": supply_high - baseline_supply,
        }

    return {"units_needed": results_units, "total_supply": results_supply}


def probability_supply_meets_demand(
    supply_samples: np.ndarray, demand_samples: np.ndarray
) -> float:
    """Probability that supply meets or exceeds demand (%)."""
    return float(np.mean(supply_samples[:, np.newaxis] >= demand_samples) * 100)


def run_and_report(
    n: int = DEFAULT_N_SAMPLES,
    params: dict | None = None,
    results_dir: str | None = None,
) -> dict:
    """
    Run full Monte Carlo, print summary (median and 90% CI), and save CSV to results folder.

    Returns dict of all summary stats for programmatic use.
    """
    user_params = params if params is not None else {}
    n = int(n)
    out_dir = results_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    sampled = _sample_all_params(n, user_params)
    stockpiled = compute_stockpiled_ppe(sampled)
    inventory_ppe = compute_inventory_ppe(sampled)
    distributed = compute_total_distributed(sampled, stockpiled, inventory_ppe)
    vital = compute_vital_workers_demand(sampled)
    people_req = compute_people_required_demand(sampled)

    def fmt(m, lo, hi):
        return f"{m:,.0f} (90% CI: {lo:,.0f} – {hi:,.0f})"

    rows = []
    summaries = {}

    # Supply: stockpiled
    for key, label in [
        ("national_excl_military", "Stockpiled PPE – national (excl. military)"),
        ("military", "Stockpiled PPE – military"),
        ("local", "Stockpiled PPE – local"),
    ]:
        m, lo, hi = _median_90ci(stockpiled[key])
        summaries[f"stockpiled_{key}"] = {"median": m, "p5": lo, "p95": hi}
        rows.append({"metric": label, "median": m, "p5": lo, "p95": hi})
        print(f"{label}: {fmt(m, lo, hi)}")

    # Inventory
    m, lo, hi = _median_90ci(inventory_ppe)
    summaries["inventory_ppe"] = {"median": m, "p5": lo, "p95": hi}
    rows.append(
        {"metric": "Total inventory PPE available", "median": m, "p5": lo, "p95": hi}
    )
    print(f"Total inventory PPE available: {fmt(m, lo, hi)}")

    # Total distributed (private sector)
    m, lo, hi = _median_90ci(distributed["private_sector"])
    summaries["total_distributed_private_sector"] = {"median": m, "p5": lo, "p95": hi}
    rows.append(
        {
            "metric": "Total distributed PPE (private sector)",
            "median": m,
            "p5": lo,
            "p95": hi,
        }
    )
    print(f"Total distributed PPE (private sector): {fmt(m, lo, hi)}")

    # Total supply
    m, lo, hi = _median_90ci(distributed["total_supply"])
    summaries["total_supply"] = {"median": m, "p5": lo, "p95": hi}
    rows.append(
        {"metric": "Total PPE supply overall", "median": m, "p5": lo, "p95": hi}
    )
    print(f"Total PPE supply overall: {fmt(m, lo, hi)}")

    # Demand: vital workers
    m, lo, hi = _median_90ci(vital["vital_workers_count"])
    summaries["vital_workers_count"] = {"median": m, "p5": lo, "p95": hi}
    rows.append({"metric": "Total vital workers", "median": m, "p5": lo, "p95": hi})
    print(f"Total vital workers: {fmt(m, lo, hi)}")

    m, lo, hi = _median_90ci(vital["vital_workers_ppe_demand"])
    summaries["vital_workers_ppe_demand"] = {"median": m, "p5": lo, "p95": hi}
    rows.append(
        {"metric": "Total PPE demand (vital workers)", "median": m, "p5": lo, "p95": hi}
    )
    print(f"Total PPE demand (vital workers): {fmt(m, lo, hi)}")

    # Demand: people required
    m, lo, hi = _median_90ci(people_req["people_required_ppe_demand"])
    summaries["people_required_ppe_demand"] = {"median": m, "p5": lo, "p95": hi}
    rows.append(
        {
            "metric": "Total PPE demand (people required to produce PPE)",
            "median": m,
            "p5": lo,
            "p95": hi,
        }
    )
    print(f"Total PPE demand (people required to produce PPE): {fmt(m, lo, hi)}")

    # Probability
    p = probability_supply_meets_demand(
        distributed["total_supply"], vital["vital_workers_ppe_demand"]
    )
    summaries["probability_supply_meets_demand_pct"] = p
    rows.append(
        {
            "metric": "P(supply >= demand for vital workers) (%)",
            "median": p,
            "p5": p,
            "p95": p,
        }
    )
    print(f"P(supply >= demand for vital workers): {p:.2f}%")

    # Save CSV
    csv_path = os.path.join(out_dir, "ppe_model_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "median", "p5", "p95"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary saved to {csv_path}")

    return summaries


if __name__ == "__main__":
    run_and_report(n=DEFAULT_N_SAMPLES)
