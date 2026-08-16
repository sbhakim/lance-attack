<p align="center">
  <img src="assets/lance-logo.png" alt="LANCE" width="460">
</p>

# LANCE

**Channel-asymmetric poisoning of temporal graph neural networks.**

LANCE is a research framework for studying training-time poisoning attacks against
temporal GNN link predictors. It re-implements the three components of the High
Impact Attack (HIA) — a surrogate model, impact-aware target selection, and hybrid
edit (injection + deletion) budgeting — inside a leakage-safe, reproducible
evaluation harness, and adds a first-order *damage-aware* edit scorer that ranks
candidate edits by their estimated marginal effect on the victim's ranking loss.
Of those three components, the ablation validates impact targeting; adaptive
budgeting is not distinguishable from a fixed split, and the injection channel does
not contribute.

The emphasis of this project is measurement. Poisoning results on temporal graphs
are easy to overstate: random negative sampling can hide deletion effects, unpaired
runs confuse training variance with attack strength, and full-history surrogates
quietly leak the future. LANCE is built so that such confounds are controlled by
construction, and so that every reported number is regenerable from the repository.

## Status and scope

LANCE is a **research prototype**, and its main result is that the two editing
channels are asymmetric in a way that tracks the *victim's architecture* rather than
what the attacker knows. Against a memory-based victim, injected noise is the
strongest attack and deleting real interactions does nothing. Against memory-free
victims, across five datasets and two victim families, the ordering reverses:
targeted deletion is the **strongest transferable attack** — roughly twice the best
undirected control — while injection transfers inconsistently and often *improves*
the victim.

Two qualifications matter. The claim we defend is **magnitude, not breadth**:
undifferentiated `random_delete` is positive in as many victim×dataset cells, so
targeting buys size of effect rather than consistency of it. And the advantage is
**budgeted**, resolvable from a 10–20% perturbation rate and absent at 5%. The
contribution is this channel finding plus a reproducible, leakage-safe harness — and
the argument that a single victim architecture measures one channel and hides the
other.

## Highlights

- **Reproducible by construction.** Deterministic kernels, recorded provenance
  (git SHA, victim, surrogate, library versions) in every artifact, and a summarizer
  that refuses to average across protocols. Seeding alone left repeats drifting by
  more than the effects being compared.
- **Knowledge regimes.** K1 (full history; the setting all reported gates use) and
  K2 (observable-prefix only, with prefix-relative or K1-matched budget).
- **Hardened evaluation.** Paired per-seed initialization, fixed clean
  destination/history pools, tie-aware MRR / Hit@*k*, TGB-style historical
  negatives, and paired *t*- and Wilcoxon tests.
- **Structurally valid perturbations.** Injected events respect source-side
  domain and timestamp/feature distributions; self-loops, duplicates, and exact
  existing events are rejected.
- **Damage-aware scorer.** A first-order influence scorer (`lance_meta`) that
  ranks edits by alignment with the gradient of the victim's ranking loss.
- **Fair controls and ablations.** `random`, `random_delete`, `random_inject`, HIA,
  an EdgeBank memorization reference, and component/targeting ablations — with a test
  asserting every attack spends its full budget, after two were found under-spending.
- **Pure PyTorch.** No DGL; runs on CPU or a single GPU.

## How it works

<p align="center">
  <img src="assets/lance-architecture.png" alt="LANCE method: a surrogate guides impact-targeted edits whose damage transfers to unseen victim architectures" width="100%">
</p>

The attack trains a memory-based surrogate, scores node impact, and spends its budget
on **impact-targeted deletions and injections**. The poisoned stream is then evaluated
on **victim architectures the attacker never sees**. The main result is that the
*deletion* channel transfers to those unseen models while injected noise does not —
and that which channel threatens a model tracks its architecture rather than what the
attacker knows. K2 (observable-prefix only) is an implemented variant; the reported
gates use K1. DT-SHIELD, a deletion-aware defense, is a future extension.

## Repository layout

```text
lance/
├── attack/     # HIA, LANCE core, meta-gradient scorer, baselines,
│               #   coverage diagnostic + coverage-maximising reference
├── models/     # TGNLite (memory), GraphMixerLite (mixer), TGATLite (attention),
│               #   EdgeBankLite (memorization reference), link predictor
├── training/   # TBPTT trainer (predict-then-update; no self-leakage)
├── eval/       # tie-aware MRR/Hit@k, historical-negative ranking
├── defense/    # DT-SHIELD components (defensive extension)
└── data/       # temporal-graph loaders (JODIE and DGB), negative sampling
configs/         # per-dataset YAML
scripts/         # train / attack / benchmark / scalability / summarize_gates
tests/           # unit and invariant tests
```

## Installation

Requires Python 3.11 and PyTorch (CUDA optional).

```bash
pip install -r requirements.txt
# optional, for exact-parity Leiden communities in the impact score:
pip install python-igraph leidenalg
```

## Data

LANCE uses public continuous-time benchmarks. The reported gates use five
JODIE/SNAP datasets — **MOOC, Wikipedia, Reddit, LastFM, Bitcoin-OTC** — and the
loader additionally reads the single-node-type graphs distributed with
[DGB](https://zenodo.org/record/7213796) (UCI, Enron, CanParl, UNtrade) via
`fmt: dgb`.

Datasets are **not** bundled. Download them and point each config's `data.root` at
the directory — by default `../Dataset` (a sibling of the repository), e.g.
`../Dataset/mooc.csv`.

## Quickstart

```bash
# Train and evaluate a clean victim
python scripts/train.py --config configs/mooc.yaml

# Run the standalone attack orchestrator (K2 builds the surrogate from the
# observable prefix only; the reported gates use K1 -- see Highlights)
python scripts/run_lance.py --config configs/mooc.yaml --knowledge k2 --ptb-rate 0.3

# Run a paired benchmark (attack × seed grid) against an unseen victim
python scripts/benchmark.py --config configs/mooc.yaml \
    --defenses none \
    --attacks none random_delete lance lance_delete \
    --seeds 0 1 --epochs 12 --ptb-rate 0.3 --hist-neg 0.7 --victim graphmixer
```

Each benchmark writes a Markdown summary and a JSON record — per-seed metrics, edit
counts, statistical tests, attack diagnostics, and the run's provenance (git SHA,
victim, surrogate, library and device versions, determinism state) — under
`artifacts/` (regenerable, and excluded from version control).

### Reproducing the reported gates

The paper's numbers come from one deterministic run set: five datasets against three
victims, on a single code revision. Determinism is on by default — repeated runs are
bit-identical, which matters because seeding alone is not sufficient (see Highlights).

```bash
# white-box (victim = surrogate family) and the two unseen victims
for V in tgnlite graphmixer tgat; do
  python scripts/benchmark.py --config configs/mooc.yaml --defenses none \
      --attacks none random random_delete random_inject hia lance \
                lance_delete lance_random_target lance_meta \
      --seeds 0 1 2 3 4 --epochs 12 --max-events 40000 \
      --ptb-rate 0.3 --hist-neg 0.7 --victim $V --out artifacts/gate_mooc_$V
done

# aggregate: per-cell table, T(a)/R(a), BH-corrected significance
python scripts/summarize_gates.py artifacts/gate_mooc_*
```

`summarize_gates.py` refuses to average across runs that differ in protocol or code
revision, so a clean run is itself evidence the cells are comparable.

Other axes use the same entry point: `--surrogate {tgnlite,graphmixer,tgat}` changes
the architecture the attacker models, `--knowledge {k1,k2}` with
`--k2-budget {prefix,matched}` selects the knowledge regime, `--victim edgebank` gives
the memorization reference, and `--ptb-rate` sweeps the budget.

## Testing

```bash
pytest -q
```

52 tests cover data loading, model contracts, ranking metrics, candidate validity,
prefix (K2) isolation and budget conventions, **budget parity across the whole attack
registry**, surrogate configurability, the coverage diagnostic, component ablations,
and diagnostic serialization. Two of them exist because they caught defects that had
already influenced reported numbers.

## Citation

```bibtex
@misc{hakim2026lance,
  author = {Safayat Bin Hakim and Wenkai Tan and Stefani Mancas and
            Sirani Mututhanthrige Perera and Houbing Herbert Song},
  title  = {Targeted Deletion Transfers, Injected Noise Does Not:
            Channel-Asymmetric Poisoning of Temporal Graph Neural Networks},
  year   = {2026},
  note   = {Research prototype; code for the LANCE attack and evaluation harness},
  url    = {https://github.com/sbhakim/lance-attack}
}
```

## License

Released under the MIT License; see [`LICENSE`](LICENSE).

## Contact

Please open an issue for project questions or contact Safayat Bin Hakim at
safayat DOT b DOT hakim AT gmail DOT com.

## Acknowledgments

Supported by the NSF Industry–University Cooperative Research Center (I/UCRC) CARTA.
