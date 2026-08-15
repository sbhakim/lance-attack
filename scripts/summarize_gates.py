"""CLI: aggregate benchmark artifacts into the tables the paper reports.

Reads any number of ``benchmark_*.json`` artifacts and emits one row per
(attack, dataset, victim) plus the two transfer summaries defined in the paper:

    T(a) = mean paired dMRR over victim x dataset cells   (magnitude)
    R(a) = fraction of cells with dMRR > 0                (reliability)

Both are reported because they can disagree: an attack can be larger on average
while another is positive in more cells, and "most transferable" means different
things under each. Rows also carry the per-cell significance count so a reader
can see how much of T(a) rests on cells that are individually resolvable.

Artifacts are grouped by the *provenance* recorded in each file; a mismatch in
code revision, victim, or determinism setting is reported rather than silently
averaged, because runs from different code are not comparable.

Example:
    python scripts/summarize_gates.py artifacts/repro_mooc_* --latex
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict


def _load(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        files = ([p] if p.endswith(".json")
                 else sorted(glob.glob(os.path.join(p, "benchmark_*.json"))))
        for f in files:
            with open(f) as fh:
                j = json.load(fh)
            j["_path"] = f
            out.append(j)
    return out


def _bh_fdr(pvals: list[float], alpha: float) -> list[bool]:
    """Benjamini-Hochberg: which p-values survive at FDR ``alpha``.

    Each attack is tested in every victim x dataset cell, so a raw count of
    ``p < alpha`` overstates how much is resolvable: at 10 cells and alpha=0.05
    one false positive is expected by chance alone. BH controls the expected
    proportion of false discoveries while staying far less brutal than Bonferroni,
    which is the right trade-off when the cells are positively dependent (the same
    attack, the same surrogate, overlapping data).
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    keep = [False] * m
    cutoff = -1
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= alpha * rank / m:
            cutoff = rank
    for rank, i in enumerate(order, start=1):
        if rank <= cutoff:
            keep[i] = True
    return keep


def _key(run: dict) -> tuple:
    """Protocol identity of a run.

    The victim is deliberately excluded: comparing victims is the point of the
    transfer study, so a victim difference is not a mismatch. What must not
    differ is everything else -- event cap, epochs, budget, negative-sampling
    mix, knowledge regime, and the code that produced it.
    """
    prov, cfg = run.get("provenance", {}), run.get("config", {})
    return (cfg.get("data", {}).get("max_events"),
            cfg.get("train", {}).get("epochs"), cfg.get("attack", {}).get("ptb_rate"),
            cfg.get("eval", {}).get("historical_neg_frac"),
            cfg.get("attack", {}).get("knowledge"),
            prov.get("git_sha"), prov.get("deterministic_algorithms"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", nargs="+", help="artifact dirs or benchmark_*.json files")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--latex", action="store_true", help="also emit a LaTeX tabular")
    args = ap.parse_args()

    runs = _load(args.artifacts)
    if not runs:
        raise SystemExit("no benchmark artifacts found")

    # cell[(dataset, victim)][attack] = (dmrr, p)
    cells: dict[tuple[str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
    protocols: dict[tuple, list[str]] = defaultdict(list)
    for r in runs:
        victim = r.get("provenance", {}).get("victim", "?")
        protocols[_key(r)].append(os.path.basename(os.path.dirname(r["_path"])))
        for row in r["rows"]:
            d = row["degradation"]
            cells[(r["dataset"], victim)][row["attack"]] = (d["mean"], d["paired_t_p"])

    if len(protocols) > 1:
        print("!! runs span more than one protocol; cells are NOT directly comparable:")
        for k, v in protocols.items():
            sha = (k[5] or "?")[:8]
            print(f"   max_events={k[0]} epochs={k[1]} ptb={k[2]} hist={k[3]} "
                  f"knowledge={k[4]} sha={sha} deterministic={k[6]}"
                  f"  <- {', '.join(sorted(set(v)))}")
        print()

    attacks = sorted({a for c in cells.values() for a in c},
                     key=lambda a: (a != "lance", a))
    keys = sorted(cells)

    print(f"{'cell':26}" + "".join(f"{a:>16}" for a in attacks))
    for k in keys:
        line = f"{k[0] + '/' + k[1]:26}"
        for a in attacks:
            if a in cells[k]:
                v, p = cells[k][a]
                line += f"{v:+.4f}{'*' if p < args.alpha else ' '}".rjust(16)
            else:
                line += f"{'--':>16}"
        print(line)

    # Coverage-reference view. oracle_delete maximises removed test-relevant
    # support, NOT damage -- measurement showed attacks exceeding it by 60%, so this
    # is a comparison against a coverage-optimal strategy, not a fraction of any
    # achievable maximum. Ratios above 100% are meaningful and expected: they say
    # the attack found damage that maximising coverage does not reach.
    if any("oracle_delete" in c for c in cells.values()):
        print(f"\n{'cell':26}{'coverage-ref dMRR':>19}   attacks vs coverage-optimal")
        for k in keys:
            orc = cells[k].get("oracle_delete", (None,))[0]
            if not orc or orc <= 0:
                print(f"{k[0] + '/' + k[1]:26}{'n/a':>14}   (no positive oracle damage)")
                continue
            parts = [f"{a}={cells[k][a][0] / orc:.0%}"
                     for a in attacks if a in cells[k] and a != "oracle_delete"]
            print(f"{k[0] + '/' + k[1]:26}{orc:+14.4f}   " + "  ".join(parts))
        print("\noracle_delete maximises removed test-relevant support, not damage."
              "\nAbove 100% means the attack beat the coverage-optimal strategy, which"
              "\nis evidence that damage is not driven by how much support is removed.")

    print(f"\n{'attack':18}{'T(a)':>10}{'R(a)':>10}{'raw sig':>10}{'BH-FDR':>9}{'best-in-cell':>14}")
    summary = []
    for a in attacks:
        vals = [cells[k][a][0] for k in keys if a in cells[k]]
        ps = [cells[k][a][1] for k in keys if a in cells[k]]
        if not vals:
            continue
        best = sum(1 for k in keys if a in cells[k]
                   and cells[k][a][0] == max(cells[k][x][0] for x in cells[k]))
        t, r = sum(vals) / len(vals), sum(v > 0 for v in vals) / len(vals)
        raw = sum(p < args.alpha for p in ps)
        bh = sum(_bh_fdr(ps, args.alpha))
        summary.append((a, t, r, raw, bh, len(vals), best))
        print(f"{a:18}{t:+10.4f}{r:>9.0%} {raw:>5}/{len(vals):<4}{bh:>4}/{len(vals):<4}{best:>10}")
    print(f"\n'raw sig' counts p < {args.alpha} per cell; 'BH-FDR' applies "
          f"Benjamini-Hochberg across that attack's cells. Report BH.")

    if args.latex:
        print("\n% --- LaTeX ---")
        print("\\begin{tabular}{l" + "c" * len(keys) + "cc}")
        print("\\toprule")
        print("Attack & " + " & ".join(k[0][:4] + "/" + k[1][:2] for k in keys)
              + " & $T(a)$ & $R(a)$ \\\\")
        print("\\midrule")
        for a, t, r, _raw, _bh, n, _best in summary:
            row = " & ".join(
                (f"${cells[k][a][0]:+.3f}$" + ("$^{*}$" if cells[k][a][1] < args.alpha else "")
                 if a in cells[k] else "--") for k in keys)
            name = "\\method" if a == "lance" else "\\texttt{" + a.replace("_", "\\_") + "}"
            print(f"{name} & {row} & ${t:+.4f}$ & {r:.0%} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()
