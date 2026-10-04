#!/usr/bin/env python
"""Build the paper's figures straight from `results/*.json`.

Nothing here hardcodes a number. If an experiment is rerun the figures change with it,
which is the only way a paper's plots and its prose stay in agreement.

    python scripts/make_figures.py --results results --outdir figures

Where a dimension has no measurement for a condition the cell is drawn as *n/a*, never as
a pass. Two of our four fault families have no natural counterpart in this pipeline, so
their ecological and decision cells are genuinely undefined, and colouring them green
would be the most misleading thing this figure could do.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

INK, GREY = "#1a1a1a", "#8c8c8c"
FAIL, PASS_, MID = "#c0392b", "#27706b", "#d9a441"
SHORT = {"query_retrieval": "retrieval", "source_degradation": "source",
         "tool_invocation": "tool", "grounding": "grounding"}
FAMILIES = ("query_retrieval", "source_degradation", "tool_invocation", "grounding")
#: Families with a natural counterpart. Everything else is n/a for V3 and V4.
COMPARABLE = ("query_retrieval", "source_degradation")
MARGIN = 0.05          # the practical margin shared by V2 and V3


def load(results: Path, name: str):
    path = results / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_latest(results: Path, *names: str):
    """Load the first available result, ordered from newest to fallback."""
    for name in names:
        value = load(results, name)
        if value is not None:
            return value
    return None


# ── Figure 1: the four dimensions disagree ───────────────────────────────────────
def figure_validity_matrix(data, out: Path):
    detector, tiers, matched, c2st, decision = data
    conditions, rows, matched_flags = [], [], []
    for tier in ("T0", "T1", "T2"):
        for family in FAMILIES:
            cond = f"{tier}:{family}"
            table = detector["by_condition"].get(cond)
            if table is None:
                continue

            # V1: how much detection is lost when the coupled features are removed.
            independence = table["decoupled"]["point"] - table["all_features"]["point"]

            # V2: prefer the matched-effect estimate, which conditions on the semantic
            # effect; fall back to the tier estimate, which does not.
            key = {"source_degradation": matched.get("source"),
                   "grounding": matched.get("grounding")}.get(family)
            if key is not None:
                pair = key["by_arm"]["decoupled"]["pairwise"]
                # The donor arm draws from a different query, so it is off-topic and
                # trivially separable; the honest contrast is between the two mechanisms
                # that draw from the same query's own rejected pool.
                same_pool = [v["point"] for k, v in pair.items() if "donor" not in k]
                artifact = float(np.mean(same_pool)) if same_pool else np.nan
            else:
                artifact = tiers["by_family"][family]["decoupled"]["macro_auroc"]["point"]

            # V3 and V4 exist only where a natural reference distribution does.
            # Query-matched C2ST: the unmatched version was confounded by the fact that
            # natural failures come only from queries the retriever failed on.
            eco = (c2st["by_condition"].get(cond, {}).get("external", {}).get("D_upper_bound")
                   if family in COMPARABLE else None)
            if family in COMPARABLE:
                decision_cell = decision["by_tier"][tier]
                stability = decision_cell.get("bootstrap_stability", {})
                agreement = stability.get("selection_agreement_probability")
                # A point argmax is not a confidence statement. Only call a stable
                # pass/fail when at least 95% of query bootstraps agree on that side;
                # otherwise show the cell as inconclusive/unstable.
                if agreement is None:
                    sel = 1.0 if decision_cell["selection_validity"] else -1.0
                elif agreement >= 0.95:
                    sel = 1.0
                elif agreement <= 0.05:
                    sel = -1.0
                else:
                    sel = 0.0
            else:
                sel = None

            conditions.append(SHORT[family])
            matched_flags.append(key is not None)
            rows.append([independence, artifact,
                         np.nan if eco is None else eco,
                         # V4 is categorical; a zero-length bar would be invisible, so
                         # both outcomes are drawn full width and told apart by colour.
                         np.nan if sel is None else sel])

    matrix = np.asarray(rows, dtype=float)
    titles = ["V1 Independence\n(detection lost\nto coupling)",
              "V2 Artifact\n(injector\nidentifiable)",
              "V3 Ecological\n(distance from\nnatural, upper bd.)",
              "V4 Decision\n(picks the right\ndetector)"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.6), sharey=True,
                             gridspec_kw={"wspace": 0.16})
    y = np.arange(len(conditions))

    specs = [
        (0, (-0.60, 0.06), lambda v: FAIL if v < -0.10 else PASS_, 0.0, "AUROC change"),
        (1, (0.45, 1.00), lambda v: FAIL if 2 * abs(v - 0.5) >= MARGIN else PASS_,
         0.5, "macro AUROC"),
        (2, (0.00, 0.62), lambda v: FAIL if v >= MARGIN else PASS_, MARGIN,
         r"$D$ upper bound"),
        (3, (0.0, 1.10), lambda v: PASS_ if v > 0 else FAIL if v < 0 else MID,
         None, "selection validity"),
    ]
    for col, (idx, xlim, colour, ref, xlabel) in enumerate(specs):
        ax = axes[col]
        values = matrix[:, idx]
        finite = np.isfinite(values)
        drawn = np.ones_like(values) if idx == 3 else values
        bars = ax.barh(y[finite], drawn[finite], height=0.62, edgecolor="white",
                       linewidth=0.5, color=[colour(v) for v in values[finite]])
        # V2 mixes two estimators: a matched-effect contrast where one exists, and the
        # weaker tier contrast otherwise. Hatch the latter so the column is not read as
        # twelve comparable numbers.
        if idx == 1:
            for bar, ok in zip(bars, np.asarray(matched_flags)[finite]):
                if not ok:
                    bar.set_hatch("///")
                    bar.set_edgecolor("white")
        if ref is not None:
            ax.axvline(ref, color=INK, lw=0.7, ls=":", zorder=0)
        ax.set_xlim(*xlim)
        ax.set_title(titles[col], pad=6)
        ax.set_xlabel(xlabel)
        for yi in y[~finite]:
            ax.text(np.mean(xlim), yi, "n/a", fontsize=6.5, style="italic",
                    color=GREY, ha="center", va="center")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[3].set_xticks([])

    axes[0].invert_yaxis()          # once only: the y-axis is shared across all four
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(conditions)
    for i in range(0, len(conditions), 4):
        axes[0].text(-0.70, i + 1.5, ["T0", "T1", "T2"][i // 4], ha="right", va="center",
                     fontsize=9, weight="bold", color=INK)

    fig.legend(handles=[Patch(facecolor=PASS_, label="passes this dimension"),
                        Patch(facecolor=FAIL, label="fails this dimension"),
                        Patch(facecolor=MID, label="inconclusive / unstable"),
                        Patch(facecolor=GREY, hatch="///", edgecolor="white",
                              label="V2 from the tier contrast (effect not held fixed)"),
                        Patch(facecolor="white", edgecolor=GREY,
                              label="n/a — no natural reference exists")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.savefig(out / "fig1_validity_matrix.pdf")
    plt.close(fig)


# ── Figure 2: how much of detection was coupling ─────────────────────────────────
def figure_coupling_collapse(detector, out: Path):
    conditions = sorted(detector["by_condition"])
    allf = [detector["by_condition"][c]["all_features"]["point"] for c in conditions]
    dec = [detector["by_condition"][c]["decoupled"]["point"] for c in conditions]
    lo = [detector["by_condition"][c]["decoupled"]["ci_low"] for c in conditions]
    hi = [detector["by_condition"][c]["decoupled"]["ci_high"] for c in conditions]

    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    x = np.arange(len(conditions))
    ax.vlines(x, dec, allf, color="#c8c8c8", lw=4, zorder=1)
    ax.scatter(x, allf, s=26, color=MID, zorder=3,
               label="all 14 features (as previously reported)")
    ax.errorbar(x, dec, yerr=[np.subtract(dec, lo), np.subtract(hi, dec)], fmt="o",
                ms=4.5, color=FAIL, lw=0.9, capsize=1.8, zorder=4,
                label="decoupled (11 features, no write-then-read path)")
    ax.axhline(0.5, color=INK, lw=0.7, ls=":", zorder=0)
    ax.text(-0.45, 0.508, "chance", fontsize=6.5, color=INK, ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c.split(':')[0]} {SHORT[c.split(':')[1]]}" for c in conditions],
                       fontsize=6.8, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_ylabel("detection AUROC")
    ax.set_ylim(0.40, 1.12)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.legend(loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.16),
              handletextpad=0.4, columnspacing=1.2)
    for i in range(len(conditions)):
        if allf[i] - dec[i] > 0.35:
            ax.annotate(f"-{allf[i] - dec[i]:.2f}", (i, (allf[i] + dec[i]) / 2),
                        xytext=(7, 0), textcoords="offset points", fontsize=6.5,
                        color=FAIL, va="center")
    fig.savefig(out / "fig2_coupling_collapse.pdf")
    plt.close(fig)


# ── Figure 3: does the fingerprint survive effect matching, and does hardening help? ──
def figure_artifact_and_ecological(tiers, matched, c2st, unmatched_c2st, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                   gridspec_kw={"wspace": 0.30})

    # (a) tier discrimination (effect NOT held fixed) against matched-effect (held fixed).
    labels, unmatched, matched_vals, nulls = [], [], [], []
    for family, key in (("source_degradation", "source"), ("grounding", "grounding")):
        run = matched.get(key)
        if run is None:
            continue
        pair = run["by_arm"]["decoupled"]["pairwise"]
        same_pool = [v["point"] for k, v in pair.items() if "donor" not in k]
        labels.append(SHORT[family])
        unmatched.append(tiers["by_family"][family]["decoupled"]["macro_auroc"]["point"])
        matched_vals.append(float(np.mean(same_pool)))
        nulls.append(run["by_arm"]["decoupled"]["permutation_null"]["max"])

    x = np.arange(len(labels))
    ax1.bar(x - 0.19, unmatched, width=0.36, color=MID, edgecolor="white",
            label="tier contrast (effect not held fixed)")
    ax1.bar(x + 0.19, matched_vals, width=0.36, color=FAIL, edgecolor="white",
            label="matched effect (same fault, different mechanism)")
    ax1.plot(x, nulls, ls="--", lw=0.9, color=INK, marker="_", ms=22,
             label="permutation null (max)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\n({'test' if l == 'source' else 'control'})" for l in labels])
    ax1.set_ylim(0.45, 0.95)
    ax1.set_ylabel("injector discrimination AUROC")
    ax1.set_title("(a) Does the fingerprint survive\nholding the fault fixed?")
    # The tallest bar reaches ~0.87, so any in-axes legend sits on the data; put it
    # underneath the panel instead.
    ax1.legend(frameon=False, fontsize=6.2, loc="upper center",
               bbox_to_anchor=(0.5, -0.22), ncol=1, handletextpad=0.5)

    # (b) distance from natural. Both the unmatched comparison (dashed) and the
    # query-matched one (solid) are shown, because the difference between them is the
    # result: the apparent divergence of the two families does not survive matching.
    for family, colour, marker in (("query_retrieval", FAIL, "o"),
                                   ("source_degradation", PASS_, "s")):
        for blob, style, alpha, tag in ((unmatched_c2st, "--", 0.45, "unmatched"),
                                        (c2st, "-", 1.0, "query-matched")):
            ys = [blob["by_condition"].get(f"{t}:{family}", {}).get("external", {})
                  .get("D_upper_bound", np.nan) for t in ("T0", "T1", "T2")]
            ax2.plot([0, 1, 2], ys, style, marker=marker, lw=1.2, ms=4, color=colour,
                     alpha=alpha, label=f"{SHORT[family]} ({tag})")
    ax2.axhline(MARGIN, color=INK, lw=0.8, ls=":")
    ax2.text(2.05, MARGIN + 0.012, "equivalence margin", fontsize=6, ha="right")
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["T0\nnaive", "T1\npublished-style", "T2\nhardened"])
    ax2.set_ylabel(r"$D$ upper bound (distance from natural)")
    ax2.set_ylim(0, 0.72)
    ax2.set_title("(b) Does hardening make faults\nmore natural-like?")
    ax2.legend(frameon=False, loc="upper left", fontsize=5.8, ncol=2,
               columnspacing=0.8, handletextpad=0.4)
    fig.savefig(out / "fig3_artifact_ecological.pdf")
    plt.close(fig)


# ── Figure 4: what a small amount of real data buys, and whether it is causal ─────
def figure_budget_and_causal(mixing, counterfactual, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5),
                                   gridspec_kw={"wspace": 0.32, "width_ratios": [1.7, 1]})

    budgets = sorted(int(b) for b in mixing["by_budget"])   # keys are strings: "160" < "20"
    styles = {"M1_natural_only": (PASS_, "-s", "natural only"),
              "M3_balanced_weight": (INK, "-^", "balanced weight"),
              "M5_matched_1x": (GREY, "-D", "matched-count"),
              "M2_naive_concat": ("#8e5aa8", "-v", "naive pooling"),
              "M4_pretrain_finetune": ("#3f6fa8", "-*", "seq. fine-tune"),
              "M0_synthetic_only": (MID, "-o", "synthetic only")}
    for key, (colour, style, label) in styles.items():
        mean = [mixing["by_budget"][str(b)][key]["mean"] for b in budgets]
        ax1.plot(budgets, mean, style, color=colour, lw=1.2, ms=3.4, label=label)
    ax1.set_xscale("log")
    ax1.set_xticks(budgets)
    ax1.set_xticklabels([str(b) for b in budgets])
    ax1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax1.set_xlabel("natural failures labelled ($n$)")
    ax1.set_ylabel("natural-fault AUROC")
    ax1.set_title("(a) No mixing protocol beats real data")
    ax1.legend(frameon=False, loc="upper left", ncol=2, fontsize=6.3,
               columnspacing=1.0, handletextpad=0.4)

    gold = counterfactual["rescue_rate_gold"]
    placebo = counterfactual["rescue_rate_placebo"]
    xs = [0, 1]
    points = [gold["point"], placebo["point"]]
    err = [[points[0] - gold["ci_low"], points[1] - placebo["ci_low"]],
           [gold["ci_high"] - points[0], placebo["ci_high"] - points[1]]]
    ax2.bar(xs, points, width=0.55, color=[PASS_, GREY], edgecolor="white")
    ax2.errorbar(xs, points, yerr=err, fmt="none", ecolor=INK, lw=0.9, capsize=3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["restore\ngold", "placebo\nswap"])
    ax2.set_ylabel("wrong answers rescued")
    ax2.set_ylim(0, 0.46)
    ax2.set_title("(b) The natural failures\nare causal")
    ax2.annotate(f"+{counterfactual['causal_effect']['point']:.3f}",
                 xy=(0.5, max(points) + 0.06), ha="center", fontsize=7.5,
                 color=INK, weight="bold")
    ax2.plot([0, 0, 1, 1], [points[0] + 0.03, max(points) + 0.05,
                            max(points) + 0.05, points[1] + 0.03],
             lw=0.8, color=INK)
    fig.savefig(out / "fig4_budget_causal.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()
    results, out = Path(args.results), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    detector = load(results, "detector_validity_v2")
    tiers = load(results, "tier_discrimination_v2")
    c2st = load(results, "c2st_matched_v2")
    unmatched = load(results, "c2st_equivalence_v2")
    decision = load_latest(results, "decision_validity_v3", "decision_validity_v2")
    mixing = load(results, "mixing_controls_v2")
    counterfactual = load(results, "counterfactual_v2")
    matched = {"source": load_latest(results, "matched_effect_source_v3",
                                      "matched_effect_source_v2",
                                      "matched_effect_source"),
               "grounding": load(results, "matched_effect_grounding")}
    needed = {"detector_validity_v2": detector, "tier_discrimination_v2": tiers,
              "c2st_matched_v2": c2st, "c2st_equivalence_v2": unmatched, "decision_validity_v2": decision,
              "mixing_controls_v2": mixing, "counterfactual_v2": counterfactual,
              "matched_effect_source": matched["source"],
              "matched_effect_grounding": matched["grounding"]}
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        raise SystemExit(f"missing results: {missing}")

    figure_validity_matrix((detector, tiers, matched, c2st, decision), out)
    figure_coupling_collapse(detector, out)
    figure_artifact_and_ecological(tiers, matched, c2st, unmatched, out)
    figure_budget_and_causal(mixing, counterfactual, out)
    for f in sorted(out.glob("fig*.pdf")):
        print(f"  {f}  {f.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
