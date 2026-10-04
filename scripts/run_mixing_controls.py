#!/usr/bin/env python
"""C2 --- is synthetic data harmful, or merely outnumbering the natural data?

The learning-curve result says twenty natural labels beat thousands of synthetic traces,
and that adding synthetic data to natural data makes things worse. That is the paper's most
quotable practical claim and it currently rests on a comparison of ~5,000 synthetic rows
against 20--160 natural ones. Domain imbalance explains the same observation: with a
class-weighted logistic fit, 5,000 rows of one domain simply dominate the objective.

Those two explanations lead to opposite advice --- "do not use synthetic data" versus "weight
it properly" --- so the claim cannot stand until they are separated. Five protocols, identical
folds, identical budgets:

    M0  synthetic only
    M1  natural only
    M2  naive concatenation                    (what the paper published)
    M3  balanced domain weight                 (each domain gets half the total weight)
    M4  synthetic pre-fit, then SGD epochs on natural only (sequential, not pooled)
    M5  matched-count synthetic augmentation   (n and 5n synthetic rows, not all of them)

    python scripts/run_mixing_controls.py --output results/mixing_controls_v2.json

Registered outcomes (docs/EXPERIMENT_PLAN.md C2): if every mixed protocol still loses to M1, the
published claim holds and strengthens. If M3, M4 or M5 recovers, the published claim was an
imbalance artifact and must be rewritten as "naive pooling harms; correct weighting does
not" --- which is more useful advice, and has to be reported even though it is less dramatic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mirage.analysis import ARMS, dedup, read_rows
from mirage.artifacts import atomic_json
from mirage.probe import raw_matrix
from mirage.validity import binary_auroc, prepare

BUDGETS = (20, 40, 80, 160)


def fit_score(train_x, train_y, test_x, weights=None):
    """Fit a standardised logistic probe and score held-out rows.

    Imputation statistics come from the training rows only. `weights` carries per-row
    sample weights, which is how M3 and M5 express domain balance.
    """
    stacked = np.vstack([train_x, test_x])
    ready = prepare(stacked, np.arange(len(train_x)))
    x_train, x_test = ready[:len(train_x)], ready[len(train_x):]

    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=5_000, class_weight="balanced")
    model.fit(scaler.transform(x_train), train_y, sample_weight=weights)
    positive = list(model.classes_).index(1)
    return model.predict_proba(scaler.transform(x_test))[:, positive]


def sequential_finetune(synth_x, synth_y, nat_x, nat_y, test_x, epochs=25, seed=11):
    """M4: fit on synthetic, then continue training on natural rows only.

    This cannot be done by warm-starting `LogisticRegression`. Its objective is convex,
    so lbfgs reaches the same optimum whatever the starting point --- a warm-started refit
    on natural data is mathematically identical to fitting natural data from scratch, and
    an earlier version of this script reported exactly that no-op as a result. Genuine
    sequential adaptation needs an optimiser whose final state depends on where it
    started, so we run SGD to convergence on synthetic and then take a bounded number of
    further passes over the natural rows.
    """
    stacked = np.vstack([synth_x, nat_x, test_x])
    ready = prepare(stacked, np.arange(len(synth_x) + len(nat_x)))
    n_s, n_n = len(synth_x), len(nat_x)
    xs, xn, xt = ready[:n_s], ready[n_s:n_s + n_n], ready[n_s + n_n:]

    scaler = StandardScaler().fit(ready[:n_s + n_n])
    xs, xn, xt = scaler.transform(xs), scaler.transform(xn), scaler.transform(xt)

    # partial_fit rejects class_weight="balanced", so the weights are computed once from
    # the synthetic labels and passed explicitly --- the same reweighting, stated up front.
    classes = np.asarray([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=synth_y)
    model = SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="optimal",
                          class_weight=dict(zip(classes.tolist(), weights)),
                          random_state=seed)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):                       # pre-train on synthetic
        order = rng.permutation(n_s)
        model.partial_fit(xs[order], synth_y[order], classes=classes)
    for _ in range(epochs):                       # then adapt on natural only
        order = rng.permutation(n_n)
        model.partial_fit(xn[order], nat_y[order])
    positive = list(model.classes_).index(1)
    return model.predict_proba(xt)[:, positive]


def domain_weights(n_a: int, n_b: int) -> np.ndarray:
    """Weights giving each domain equal total mass regardless of its row count."""
    return np.concatenate([np.full(n_a, 0.5 / max(n_a, 1)),
                           np.full(n_b, 0.5 / max(n_b, 1))]) * (n_a + n_b)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--train-experiment", default="E3_positive_control_v2")
    parser.add_argument("--test-experiment", default="E4_fingerprint_v2")
    parser.add_argument("--arm", default="decoupled", choices=list(ARMS))
    parser.add_argument("--tier", default="T1")
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts).resolve()
    columns = ARMS[args.arm]

    train_rows = dedup(read_rows(root, args.train_experiment))
    synth = [r for r in train_rows if str(r.get("fault_tier")) == args.tier
             or str(r.get("fault_source")) == "clean"]
    synth_x = raw_matrix(synth, columns)
    synth_y = np.asarray([int(str(r.get("fault_source")) != "clean") for r in synth])

    clean = [r for r in dedup(read_rows(root, args.test_experiment))
             if str(r.get("fault_source", "")) == "clean"]
    nat_x = raw_matrix(clean, columns)
    nat_y = np.asarray([int(bool(r.get("natural_retrieval_failure"))) for r in clean])
    nat_q = np.asarray([str(r["query_id"]) for r in clean])
    positives = np.unique(nat_q[nat_y == 1])
    negatives = np.unique(nat_q[nat_y == 0])
    print(f"synthetic pool {len(synth)} rows; natural pool {len(clean)} traces "
          f"({len(positives)} positive queries)")

    rng = np.random.default_rng(args.seed)
    protocols = ["M0_synthetic_only", "M1_natural_only", "M2_naive_concat",
                 "M3_balanced_weight", "M4_pretrain_finetune",
                 "M5_matched_1x", "M5_matched_5x"]
    result = {"analysis": "synthetic_natural_mixing_controls", "arm": args.arm,
              "tier": args.tier, "draws": args.draws, "budgets": list(BUDGETS),
              "protocols": protocols, "by_budget": {}}

    for budget in BUDGETS:
        if budget > len(positives):
            continue
        scores = {p: [] for p in protocols}
        for _ in range(args.draws):
            pos = rng.choice(positives, size=budget, replace=False)
            neg = rng.choice(negatives, size=min(budget * 2, len(negatives)), replace=False)
            labelled = set(pos) | set(neg)
            train_mask = np.isin(nat_q, list(labelled))
            test_mask = ~train_mask
            if len(set(nat_y[train_mask])) < 2 or len(set(nat_y[test_mask])) < 2:
                continue

            nx, ny = nat_x[train_mask], nat_y[train_mask]
            tx, ty = nat_x[test_mask], nat_y[test_mask]

            scores["M0_synthetic_only"].append(
                binary_auroc(ty, fit_score(synth_x, synth_y, tx)))
            scores["M1_natural_only"].append(binary_auroc(ty, fit_score(nx, ny, tx)))

            pooled_x = np.vstack([synth_x, nx])
            pooled_y = np.concatenate([synth_y, ny])
            scores["M2_naive_concat"].append(
                binary_auroc(ty, fit_score(pooled_x, pooled_y, tx)))
            scores["M3_balanced_weight"].append(binary_auroc(ty, fit_score(
                pooled_x, pooled_y, tx, weights=domain_weights(len(synth_y), len(ny)))))
            scores["M4_pretrain_finetune"].append(binary_auroc(ty, sequential_finetune(
                synth_x, synth_y, nx, ny, tx, seed=args.seed)))

            for name, multiple in (("M5_matched_1x", 1), ("M5_matched_5x", 5)):
                take = min(len(synth_y), budget * 3 * multiple)
                pick = rng.choice(len(synth_y), size=take, replace=False)
                scores[name].append(binary_auroc(ty, fit_score(
                    np.vstack([synth_x[pick], nx]),
                    np.concatenate([synth_y[pick], ny]), tx)))

        entry = {p: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
                     "n_draws": int(np.isfinite(v).sum())}
                 for p, v in scores.items() if v}
        # Draws share folds across protocols, so the difference is paired and its
        # spread is far tighter than the marginal SDs suggest. Comparing marginal
        # means alone cannot say whether a protocol genuinely beats natural-only.
        base = np.asarray(scores["M1_natural_only"], dtype=float)
        for p in entry:
            delta = np.asarray(scores[p], dtype=float) - base
            ok = np.isfinite(delta)
            lo, hi = (np.percentile(delta[ok], [2.5, 97.5]) if ok.any() else (np.nan,) * 2)
            entry[p]["paired_vs_natural"] = {
                "delta": float(np.nanmean(delta)), "ci_low": float(lo),
                "ci_high": float(hi), "wins": int((delta[ok] > 0).sum()),
                "n": int(ok.sum()),
                "beats_natural": bool(lo > 0), "loses_to_natural": bool(hi < 0)}
        # Every protocol is judged against natural-only, which is the thing a
        # practitioner would fall back to if told to ignore synthetic data entirely.
        for p in entry:
            entry[p]["delta_vs_natural_only"] = (entry[p]["mean"]
                                                 - entry["M1_natural_only"]["mean"])
        result["by_budget"][str(budget)] = entry
        print(f"\nn={budget}")
        for p in protocols:
            if p in entry:
                print(f"  {p:24s} {entry[p]['mean']:.3f} +- {entry[p]['sd']:.3f}"
                      f"   vs natural-only {entry[p]['delta_vs_natural_only']:+.3f}")

    largest = str(BUDGETS[-1])
    final = result["by_budget"].get(largest, {})
    mixed = [p for p in protocols if p not in ("M0_synthetic_only", "M1_natural_only")]
    recovered = [p for p in mixed
                 if final.get(p, {}).get("paired_vs_natural", {}).get("beats_natural")]
    result["verdict"] = {
        "budget": largest,
        "protocols_matching_natural_only": recovered,
        "interpretation": ("synthetic information transfers negatively under every mixing "
                           "protocol tested" if not recovered else
                           "naive pooling harms, but correct domain weighting recovers it; "
                           "the published claim was an imbalance artifact")}
    atomic_json(Path(args.output), result)

    print("\n" + "=" * 72)
    print(f"VERDICT at n={largest}: {result['verdict']['interpretation']}")
    if recovered:
        print(f"  recovered: {', '.join(recovered)}")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
