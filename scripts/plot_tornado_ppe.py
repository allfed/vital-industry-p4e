import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.ppe_model import PARAMETERS, RESULTS_DIR, run_one_way_sensitivity


def _effect_size(delta_low: float, delta_high: float) -> float:
    return max(abs(delta_low), abs(delta_high))


def _prepare_tornado_data(
    sensitivity: Dict[str, Dict[str, float]],
    top_n: int,
) -> Tuple[List[str], np.ndarray, np.ndarray, float]:
    names = []
    lows = []
    highs = []
    baselines = []

    for name, stats in sensitivity.items():
        names.append(name)
        lows.append(stats["delta_low"])
        highs.append(stats["delta_high"])
        baselines.append(stats["baseline"])

    if not names:
        return [], np.array([]), np.array([]), 0.0

    baselines_arr = np.array(baselines)
    # All baselines are identical per output
    baseline_value = float(baselines_arr[0])

    delta_low_arr = np.array(lows)
    delta_high_arr = np.array(highs)

    effects = np.array(
        [_effect_size(dl, dh) for dl, dh in zip(delta_low_arr, delta_high_arr)]
    )
    order = np.argsort(effects)[::-1][:top_n]

    ordered_names = [names[i] for i in order]
    ordered_low = delta_low_arr[order]
    ordered_high = delta_high_arr[order]

    # Convert to millions of units for plotting
    scale = 1e6
    ordered_low_scaled = (baseline_value + ordered_low) / scale
    ordered_high_scaled = (baseline_value + ordered_high) / scale
    baseline_scaled = baseline_value / scale

    return ordered_names, ordered_low_scaled, ordered_high_scaled, baseline_scaled


def _label_for_param(name: str) -> str:
    # Template mapping for nicer labels; update these as needed.
    pretty_names = {
        "proportion_vital_vs_essential": "% Vital vs essential workers",
        "proportion_needing_p4e": "% Vital workers needing PPE",
        "units_p4e_per_worker": "Units PPE per vital worker",
        "all_essential_workers": "Number of essential workers",
        "military_m50_stockpile": "Military M50 stockpile size",
        "sns_ehmrs": "SNS EHMRs stockpile size",
        # "papr_spending": "TEMPLATE: PAPR spending",
        # "papr_unit_cost": "TEMPLATE: PAPR unit cost",
        # "ratio_purchase_to_stockpile": "TEMPLATE: purchase-to-stockpile ratio",
        # "proportion_national_paprs": "TEMPLATE: proportion national PAPRs",
        "paprs_stock_on_hand": "PAPRs in hospitals",
        # "proportion_facilities_with_elastomerics": "TEMPLATE: facilities with elastomerics",
        # "elastomerics_needed_flu_pandemic": "TEMPLATE: elastomerics needed (flu pandemic)",
        # "actual_facilities_stocked": "TEMPLATE: actual facilities stocked",
        # "us_papr_market": "TEMPLATE: US PAPR market",
        # "papr_cost_inventory": "TEMPLATE: PAPR cost (inventory)",
        # "suitability": "TEMPLATE: suitability",
        "days_on_hand": "Days PPE on hand (inventory)",
        "n95_sales_pre_covid": "N95 sales pre-COVID",
        "elastomeric_sales_vs_n95": "Elastomeric sales vs N95 sales",
        "elastomeric_unit_cost": "Elastomeric unit cost",
        "proportion_suitable_filters": "% Elastomerics with suitable filters",
        # "papr_private_sector_employees": "TEMPLATE: PAPR private-sector employees",
        # "paprs_per_papr_employee": "TEMPLATE: PAPRs per PAPR employee",
        # "non_powered_private_sector_employees": "TEMPLATE: non-powered private-sector employees",
        # "ehmr_usage_orgs": "TEMPLATE: EHMR usage orgs",
        "employee_org_usage_ratio": "% Employees using PPE*",
        "elastomerics_per_elastomeric_employee": "Elastomerics per employee*",
        # "efmr_usage_orgs": "TEMPLATE: EFMR usage orgs",
        # "law_enforcement_total": "TEMPLATE: law enforcement total",
        # "energy_workers_total": "TEMPLATE: energy workers total",
        # "proportion_needed_energy_workers": "TEMPLATE: proportion needed energy workers",
        # "usps_non_mail_carriers_total": "TEMPLATE: USPS non-mail carriers total",
        # "usps_mail_carriers_total": "TEMPLATE: USPS mail carriers total",
        # "proportion_needed_non_mail": "TEMPLATE: proportion needed non-mail",
        # "proportion_needed_mail": "TEMPLATE: proportion needed mail",
        # "telecomms_total": "TEMPLATE: telecomms total",
        # "proportion_needed_telecomms": "TEMPLATE: proportion needed telecomms",
        # "ppe_workers_total": "TEMPLATE: PPE workers total",
        # "proportion_needed_ppe_workers": "TEMPLATE: proportion needed PPE workers",
    }

    if name in pretty_names:
        return pretty_names[name]
    # Fallback: simple human-readable label from parameter name
    return name.replace("_", " ")


def plot_tornado(top_n: int = 10) -> None:
    sensitivity = run_one_way_sensitivity()

    units_sens = sensitivity["units_needed"]
    supply_sens = sensitivity["total_supply"]

    (
        names_units,
        low_units,
        high_units,
        baseline_units,
    ) = _prepare_tornado_data(units_sens, top_n)
    (
        names_supply,
        low_supply,
        high_supply,
        baseline_supply,
    ) = _prepare_tornado_data(supply_sens, top_n)

    plt.style.use(
        "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
    )

    plt.rcParams.update(
        {
            "font.size": 11.5,
            "legend.fontsize": 13,
        }
    )

    # Use different subplot heights proportional to number of bars so bar
    # thickness looks similar in both panels.
    n_units = max(len(names_units), 1)
    n_supply = max(len(names_supply), 1)
    fig, (ax_a, ax_b) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        dpi=300,
        gridspec_kw={"height_ratios": [n_units, n_supply]},
    )

    # (a) Units needed
    y_pos_a = np.arange(len(names_units))
    for i, (name, yl, yh) in enumerate(zip(names_units, low_units, high_units)):
        left = min(yl, yh)
        width = abs(yh - yl)
        ax_a.barh(i, width, left=left, color="#F8766D", alpha=0.8)

        # annotate low/high input values
        low_input = PARAMETERS[name]["low"]
        high_input = PARAMETERS[name]["high"]
        x_low = min(yl, yh)
        x_high = max(yl, yh)

        # Small horizontal gap so text does not collide with bars/labels
        x_min, x_max = ax_a.get_xlim()
        gap = 0.01 * (x_max - x_min) if x_max > x_min else 0.0

        ax_a.text(
            x_low - gap,
            i,
            f"{low_input:,.2g}",
            va="center",
            ha="right",
            fontsize=8,
        )
        ax_a.text(
            x_high + gap,
            i,
            f"{high_input:,.2g}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax_a.axvline(baseline_units, color="darkgray", linestyle="--", linewidth=1)
    ax_a.set_yticks(y_pos_a)
    ax_a.set_yticklabels([_label_for_param(n) for n in names_units])
    # Add padding so y-axis labels do not collide with low-value annotations
    ax_a.tick_params(axis="y", which="major", pad=30)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Units of PPE (millions)", fontweight="bold")
    ax_a.xaxis.set_label_coords(0.35, -0.2)

    # (b) Total rapidly accessible P4E (ALL)
    y_pos_b = np.arange(len(names_supply))
    for i, (name, yl, yh) in enumerate(zip(names_supply, low_supply, high_supply)):
        left = min(yl, yh)
        width = abs(yh - yl)
        ax_b.barh(i, width, left=left, color="#00BFC4", alpha=0.8)

        low_input = PARAMETERS[name]["low"]
        high_input = PARAMETERS[name]["high"]
        x_low = min(yl, yh)
        x_high = max(yl, yh)

        x_min, x_max = ax_b.get_xlim()
        gap = 0.01 * (x_max - x_min) if x_max > x_min else 0.0

        ax_b.text(
            x_low - gap,
            i,
            f"{low_input:,.2g}",
            va="center",
            ha="right",
            fontsize=8,
        )
        ax_b.text(
            x_high + gap,
            i,
            f"{high_input:,.2g}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax_b.axvline(baseline_supply, color="darkgray", linestyle="--", linewidth=1)
    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels([_label_for_param(n) for n in names_supply])
    ax_b.tick_params(axis="y", which="major", pad=30)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Units of PPE (millions)", fontweight="bold")
    ax_b.xaxis.set_label_coords(0.35, -0.05)

    # Place subplot titles in figure coordinates, aligned with left edge of label text
    fig.text(
        0.02,
        0.98,
        "(a)",
        ha="left",
        va="top",
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.70,
        "(b)",
        ha="left",
        va="top",
        fontweight="bold",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), h_pad=4)

    # Add vertical space *between* subplots
    # fig.subplots_adjust(hspace=0.4, top=0.95)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "p4e_tornado_sensitivity.png")
    fig.savefig(out_path, dpi=300)


if __name__ == "__main__":
    plot_tornado()
