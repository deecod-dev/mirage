"""Run external detectors over MIRAGE traces, on identical data.

Cross-paper AUROC comparison is not available to us: published numbers come from
different datasets, different fault definitions and different label balances, so a
table putting our 0.975 next to someone's 0.78 says nothing. The only defensible
comparison is to run their detector on *our* traces. That is what this does.

Baselines
---------
``hhem``      Vectara HHEM — a cross-encoder built for exactly this pairing
              (premise = evidence the generator saw, hypothesis = its answer).
``nli``       The DeBERTa-MNLI scoring function SelfCheckGPT-NLI uses, applied to
              (evidence, answer). **This is not SelfCheckGPT.** The real method scores
              consistency across N stochastic samples of the same generation, and our
              pipeline decodes greedily so those samples do not exist. Running the real
              method would mean regenerating every answer N times with temperature > 0.
              Reported under its own name so the paper never claims to have run
              SelfCheckGPT when it did not.
``judge``     LLM-as-judge on the same traces, using the pipeline's own generator.
``mirage``    Our 14-metric suite, for reference in the same table.

Symmetry ("comparing metrics-with-oracle against a judge-without-oracle
is confounded")
---------------------------------------------------------------------------------
Every method is evaluated two ways:

* **unsupervised** — its raw score against the label, which is how single-score
  detectors are normally deployed;
* **probed** — its score(s) fed to the same logistic probe under the same
  query-disjoint splits MIRAGE gets, so a 14-feature suite is not silently compared
  against a 1-feature detector that was denied any fitting.

Scores are cached per trace, so re-running to add a baseline never recomputes an old one.

    python scripts/run_baselines.py --experiment E4_fingerprint_v2 \
        --baselines hhem nli mirage --output results/baselines.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from mirage.analysis import ARMS, FEATURES, dedup, read_rows
from mirage.artifacts import atomic_json
from mirage.pipeline import evidence_block
from mirage.probe import new_probe, raw_matrix

HHEM_MODEL = "vectara/hallucination_evaluation_model"
#: The scoring model SelfCheckGPT-NLI itself uses. Two-class, unnamed labels.
NLI_MODEL = "potsawee/deberta-v3-large-mnli"
#: A strong, properly-labelled general NLI model, as a second external reference.
NLI_LARGE_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

JUDGE_SYSTEM = (
    "You audit retrieval-augmented answers. Given a question, the evidence the system "
    "actually used, and its answer, rate how likely it is that something went wrong "
    "anywhere in the pipeline — wrong evidence retrieved, evidence corrupted, a tool "
    "misused, or an unsupported claim. Return JSON only: {\"fault_likelihood\": <0-100>}."
)


# ── score caching ────────────────────────────────────────────────────────────
def cache_path(root: Path, experiment: str, baseline: str) -> Path:
    return root / experiment / "baseline_scores" / f"{baseline}.json"


def load_cache(path: Path) -> dict[str, float]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def trace_key(row: dict[str, Any]) -> str:
    return f"{row['query_id']}|{row.get('fault_source')}|{row.get('injector_seed')}"


def score_with_cache(rows: Sequence[dict], path: Path, compute: Callable[[list[dict]], list[float]],
                     ) -> tuple[np.ndarray, float]:
    """Return (scores aligned to rows, seconds per newly-scored trace)."""
    cached = load_cache(path)
    missing = [r for r in rows if trace_key(r) not in cached]
    seconds = 0.0
    if missing:
        started = time.monotonic()
        values = compute(missing)
        seconds = (time.monotonic() - started) / len(missing)
        cached.update({trace_key(r): float(v) for r, v in zip(missing, values)})
        atomic_json(path, cached)
    return np.asarray([cached[trace_key(r)] for r in rows], dtype=float), seconds


# ── baselines ────────────────────────────────────────────────────────────────
def pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    available, index = max(free)
    if available < 3 * 1024 ** 3:
        print(f"  no GPU has 3GB free (best: {available / 1024 ** 3:.1f}GB) — using CPU")
        return "cpu"
    return f"cuda:{index}"


def nli_indices(config) -> tuple[int, int]:
    """Locate the entailment and contradiction heads.

    NLI checkpoints disagree about label order and some ship none at all: the model
    SelfCheckGPT uses reports {0: LABEL_0, 1: LABEL_1}. Guessing wrong silently
    inverts the detector, so resolve by name where possible and fall back only to the
    documented two-class convention.
    """
    names = {i: str(v).lower() for i, v in config.id2label.items()}
    entail = next((i for i, v in names.items() if "entail" in v), None)
    contra = next((i for i, v in names.items() if "contradict" in v), None)
    if entail is not None and contra is not None:
        return entail, contra
    if config.num_labels == 2:
        return 0, 1          # SelfCheckGPT's convention: [entailment, contradiction]
    raise ValueError(f"cannot resolve NLI labels from {config.id2label}")


def hhem_scorer(device: str, batch_size: int):
    """Higher score = more likely faulty (HHEM returns a consistency probability).

    HHEM ships custom remote code written against transformers 4.x. On transformers 5
    it fails with `all_tied_weights_keys`. Downgrading is not worth breaking the main
    pipeline for, so this reports clearly and the NLI baselines stand in.
    """
    from transformers import AutoModelForSequenceClassification
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            HHEM_MODEL, trust_remote_code=True).to(device).eval()
    except AttributeError as exc:
        raise RuntimeError(
            "HHEM's remote code is incompatible with the installed transformers "
            f"({exc}). Score it in a side venv with transformers<5 if you need it; "
            "the nli / nli-large baselines run here unaffected.") from exc

    def compute(rows: list[dict]) -> list[float]:
        pairs = [(evidence_block(r["selected_context"]), r["answer"]) for r in rows]
        out: list[float] = []
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            consistency = model.predict(chunk)
            out.extend(1.0 - float(v) for v in np.asarray(consistency).reshape(-1))
        return out
    return compute


def nli_scorer(device: str, batch_size: int, model_name: str = NLI_MODEL):
    """SelfCheckGPT-NLI's scoring function: P(contradiction) over {entail, contra}."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    entail, contra = nli_indices(model.config)

    def compute(rows: list[dict]) -> list[float]:
        out: list[float] = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            encoded = tokenizer([evidence_block(r["selected_context"]) for r in chunk],
                                [r["answer"] for r in chunk],
                                return_tensors="pt", padding=True, truncation=True,
                                max_length=512).to(device)
            with torch.inference_mode():
                probabilities = torch.softmax(model(**encoded).logits, dim=-1)
            pair = probabilities[:, [entail, contra]]
            out.extend((pair[:, 1] / pair.sum(dim=1).clamp(min=1e-9)).cpu().tolist())
        return out
    return compute


def judge_scorer(config_path: str, device: str):
    """LLM-as-judge using the pipeline's own generator, so the comparison is symmetric."""
    import os
    # The pipeline pins its generator to `device_map={"": 0}` — index 0 of the *visible*
    # devices. Restricting visibility here is what makes that land on the requested card;
    # without it the judge silently loads onto physical GPU 0.
    if device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    elif device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from mirage.config import load_config
    from mirage.pipeline import RealAgenticRAG
    agent = RealAgenticRAG(load_config(config_path))

    def compute(rows: list[dict]) -> list[float]:
        prompts = [(JUDGE_SYSTEM,
                    f"Question: {r['question']}\n\nEvidence:\n{evidence_block(r['selected_context'])}"
                    f"\n\nAnswer: {r['answer']}") for r in rows]
        out = []
        for text in agent.chat_many(prompts, max_new_tokens=48):
            try:
                out.append(float(RealAgenticRAG._json_object(text)["fault_likelihood"]) / 100.0)
            except Exception:
                out.append(0.5)          # unparseable verdict is no evidence either way
        return out
    return compute


# ── evaluation ───────────────────────────────────────────────────────────────
def query_splits(rows: Sequence[dict], repeats: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    queries = np.unique([str(r["query_id"]) for r in rows])
    owner = np.asarray([str(r["query_id"]) for r in rows])
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(repeats):
        shuffled = rng.permutation(queries)
        half = set(shuffled[:len(shuffled) // 2])
        mask = np.asarray([q in half for q in owner])
        splits.append((np.where(mask)[0], np.where(~mask)[0]))
    return splits


def prepare(matrix: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Impute with the TRAINING median and append definedness indicators.

    Imputing from the whole set before splitting would leak test statistics into the
    fit — small for a median, but it is exactly the kind of shortcut this project
    exists to complain about, so the medians are recomputed per split.
    """
    with np.errstate(all="ignore"):
        medians = np.nan_to_num(np.nanmedian(matrix[train_idx], axis=0), nan=0.0)
    defined = np.isfinite(matrix).astype(float)
    filled = np.where(np.isfinite(matrix), matrix, medians)
    return np.hstack([filled, defined]) if defined.min() == 0 else filled


def evaluate(columns: np.ndarray, labels: np.ndarray, splits) -> dict[str, float]:
    """Raw AUROC for a single score; probed AUROC under query-disjoint splits."""
    matrix = columns if columns.ndim == 2 else columns.reshape(-1, 1)
    result: dict[str, float] = {}
    if matrix.shape[1] == 1 and np.isfinite(matrix).all():
        try:
            raw = roc_auc_score(labels, matrix[:, 0])
            result["auroc_unsupervised"] = float(max(raw, 1 - raw))
        except ValueError:
            pass
    scores = []
    for train_idx, test_idx in splits:
        if len(set(labels[train_idx])) < 2 or len(set(labels[test_idx])) < 2:
            continue
        prepared = prepare(matrix, train_idx)
        model = new_probe()
        model.fit(prepared[train_idx], labels[train_idx])
        positive = list(model.classes_).index(1)
        scores.append(roc_auc_score(labels[test_idx],
                                    model.predict_proba(prepared[test_idx])[:, positive]))
    if scores:
        result["auroc_probed"] = float(np.mean(scores))
        result["auroc_probed_sd"] = float(np.std(scores))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--baselines", nargs="+",
                        default=["nli", "nli-large", "mirage", "trivial"],
                        choices=["hhem", "nli", "nli-large", "judge", "mirage", "trivial"])
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda:N")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--judge-config", default="configs/experiments/E4_fingerprint.yaml")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--limit", type=int, default=None, help="score only N traces (smoke test)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts).resolve()
    rows = dedup(read_rows(root, args.experiment))
    if args.limit:
        keep = {str(r["query_id"]) for r in rows}
        keep = set(sorted(keep)[:max(1, args.limit // 13)])
        rows = [r for r in rows if str(r["query_id"]) in keep]
    rows = [r for r in rows if r.get("selected_context") and r.get("answer")]
    print(f"{len(rows)} traces, {len({r['query_id'] for r in rows})} queries")

    needs_gpu = {"hhem", "nli", "nli-large", "judge"} & set(args.baselines)
    device = pick_device(args.device) if needs_gpu else "cpu"
    if needs_gpu:
        print(f"device: {device}")

    columns: dict[str, np.ndarray] = {}
    timing: dict[str, float] = {}
    for name in args.baselines:
        print(f"\nscoring {name} ...", flush=True)
        try:
            if name in ("mirage", "trivial"):
                # Metrics are already on the trace; 112 ms/trace measured separately.
                features = FEATURES if name == "mirage" else ARMS["trivial_baselines"]
                columns[name] = raw_matrix(rows, features)
                timing[name] = 0.112     # measured on GPU; see the caveat printed below
                print("  (precomputed on the trace)")
                continue
            compute = {"hhem": lambda: hhem_scorer(device, args.batch_size),
                       "nli": lambda: nli_scorer(device, args.batch_size, NLI_MODEL),
                       "nli-large": lambda: nli_scorer(device, args.batch_size, NLI_LARGE_MODEL),
                       "judge": lambda: judge_scorer(args.judge_config, device)}[name]()
            values, seconds = score_with_cache(
                rows, cache_path(root, args.experiment, name), compute)
            columns[name], timing[name] = values, seconds
            print(f"  {seconds * 1000:.0f} ms/trace" if seconds else "  (all cached)")
        except Exception as exc:
            print(f"  SKIPPED — {type(exc).__name__}: {exc}")

    clean = [r for r in rows if r.get("fault_source") == "clean"]
    conditions = sorted({str(r["fault_source"]) for r in rows} - {"clean"})
    table: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        subset = clean + [r for r in rows if str(r.get("fault_source")) == condition]
        index = {trace_key(r): i for i, r in enumerate(rows)}
        picked = np.asarray([index[trace_key(r)] for r in subset])
        labels = np.asarray([int(r.get("fault_source") != "clean") for r in subset])
        splits = query_splits(subset, args.repeats, args.seed)
        table[condition] = {
            name: evaluate(values[picked], labels, splits)
            for name, values in columns.items()}

    result = {"analysis": "external_baseline_comparison", "experiment": args.experiment,
              "n_traces": len(rows), "n_queries": len({r["query_id"] for r in rows}),
              "device": device, "repeats": args.repeats,
              "seconds_per_trace": timing, "by_condition": table,
              "timing_comparable": device != "cpu",
              "note": "'nli' is the SelfCheckGPT-NLI scoring function applied to "
                      "(evidence, answer); it is NOT SelfCheckGPT, which needs sampled "
                      "generations this pipeline does not produce."}
    atomic_json(Path(args.output), result)

    names = list(columns)
    print(f"\n{'condition':26s} " + "  ".join(f"{n:>16s}" for n in names))
    print(f"{'':26s} " + "  ".join(f"{'raw / probed':>16s}" for _ in names))
    for condition, per in table.items():
        cells = []
        for name in names:
            entry = per.get(name, {})
            raw = entry.get("auroc_unsupervised")
            probed = entry.get("auroc_probed")
            cells.append(f"{raw:.3f}/{probed:.3f}" if raw is not None and probed is not None
                         else f"{'—':>5}/{probed:.3f}" if probed is not None else f"{'n/a':>16}")
        print(f"{condition:26s} " + "  ".join(f"{c:>16s}" for c in cells))
    print("\nseconds per trace: " + ", ".join(f"{k}={v * 1000:.0f}ms" for k, v in timing.items()))
    if device == "cpu" and {"mirage", "trivial"} & set(timing):
        print("  CAVEAT: mirage's 112 ms is a GPU measurement while the baselines just ran "
              "on CPU. Do not quote this as a speed comparison — re-run with --device cuda:N "
              "so every method is timed on the same hardware.")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
