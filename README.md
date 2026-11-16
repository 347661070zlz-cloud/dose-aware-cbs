# Dose-aware resource-constrained CBS for multi-UAV path planning

Research code for a manuscript in preparation.

**This repository contains code only.** Neither the manuscript (text, figures, PDF) nor the result
data is published here. The scripts regenerate every CSV from scratch; they write into
`code/results/thesis/`, which is git-ignored.

---

## What the code does

**Problem.** Multi-agent path finding (MAPF) on a 3-D voxel grid, where each agent accumulates a
*resource* (smoke dose) along its route and every agent must stay below a per-agent cumulative cap.
The cap is a feasibility condition, not a cost term: a route that exceeds it is unusable.

**Method.** A CBS-family high level is left untouched. The low-level single-agent search is replaced
by a label-setting search that treats dose as a second resource dimension:

* dose is accumulated exactly along the path (no approximation is carried into the result);
* a bucketed index (`K` intervals) is used *only* for the Pareto-dominance test;
* any label whose accumulated dose exceeds the budget is pruned, so the search cannot return an
  over-exposed route.

Because the cap is enforced during the search rather than checked afterwards, the planner returns
either a compliant plan or no plan. Bucketed dominance makes the search sound but not complete;
the code also supports exact-dose dominance, which is slower and complete.

**Evaluation setting.** Public 3D BAG building data for one 100 × 100 m urban window, with a
synthetic advected smoke plume, a corridor that every agent must cross, and team sizes of 6–12.

---

## Layout

```
code/    planner and one script per experiment
```

| Path | Contents |
|---|---|
| `code/lowlevel.py` | dose-aware label-setting search (exact accumulation, bucketed dominance) |
| `code/highlevel.py` | CBS / ECBS constraint tree; dose mode passed to every low-level call |
| `code/cost.py`, `code/config.py` | cost terms, hard constraints, all parameters |
| `code/environment.py` | scene generation (buildings, no-fly wall, plume, wind) |
| `code/bag_data.py` | 3D BAG ingestion and voxel cache |
| `code/methods.py` | method definitions (CBS, ECBS, dose-aware variants) |
| `code/thesis_experiments.py` | main experiment driver |
| remaining scripts | one experiment each (see below) |

---

## Requirements

* Python 3.10+ with **numpy** (matplotlib only for the figure scripts)
* No GPU, no MILP/CP solver, no external service
* Network access **once**, to build the building-geometry cache (see step 1)

Per-scenario planning is capped at 3 s by default, so a full 80-scenario group takes several
minutes; the exact-search diagnostics are the slowest scripts.

---

## Running

All commands run from `code/`.

### 1. Build the geometry cache (once, needs network)

```bash
python bag_data.py --download   # writes data/bag_voxels.json under code/
python bag_data.py --info       # sanity check: scene statistics
```

The cache is derived from the public 3D BAG WFS endpoint; it is not shipped with this repository.
Once it exists, every experiment runs offline.

### 2. Run the experiments

Outputs are written to `code/results/thesis/`; the `--bag` flag appends `_bag` to the file name
(e.g. `dose_main_bag.csv`).

```bash
# Main comparison: 80 scenarios (6/8/10/12 agents x 20 seeds), four methods
python thesis_experiments.py --group dose_main --bag --t-budget 3.0 --dose-budget 2.88

# Cap sweep: one group per cap level
python thesis_experiments.py --group ecbs --bag --t-budget 3.0 --dose-budget 2.88

# Ablation: bucket count and the dose lower bound
python thesis_experiments.py --group dose --bag --t-budget 3.0 --dose-budget 2.88
python thesis_experiments.py --group dose --bag --t-budget 3.0 --dose-budget 2.88 --dose-buckets 8
python thesis_experiments.py --group dose --bag --t-budget 3.0 --dose-budget 2.88 --dose-buckets 32
python thesis_experiments.py --group dose --bag --t-budget 3.0 --dose-budget 2.88 --no-dose-prune

# Re-solving violating scenarios with a larger time budget
python detour_experiment.py --budget 2.88 --solve-budget 15.0

# Post-hoc filtering baselines (reject / re-solve after the fact)
python filter_baseline.py --budget 2.88 --solve-budget 3.0

# EECBS-style variant under the same cap
python run_eecbs.py

# Window census and the two contrasting windows
python window_survey.py
python window_robustness.py
```

The diagnostics below read the CSVs produced above, so run them afterwards:

```bash
python recalibrate_cap.py    # out-of-sample standardisation of the cap
python recal_layer.py        # re-run the layer at that exogenous cap
python horizon_probe.py      # relax the search horizon on the refusals
python dominance_probe.py    # per-agent feasibility, exact vs bucketed dominance
python cert_feasibility.py   # single-agent relaxation
python exact_pilot.py        # exact-dose joint search, three pilots
```

### 3. Figures

```bash
python make_paper_figures.py   # writes into ../figures (git-ignored)
python make_framework_fig.py
```

---

## What each script writes

| Output | Written by |
|---|---|
| `dose_main_bag.csv` | `thesis_experiments.py --group dose_main` |
| `ecbs_B<cap>.csv` | `thesis_experiments.py --group ecbs` |
| `dose_ab_*.csv` | `thesis_experiments.py --group dose` (four variants) |
| `detour.csv`, `detour_60s.csv` | `detour_experiment.py` |
| `filter_baseline.csv` | `filter_baseline.py` |
| `eecbs.csv` | `run_eecbs.py` |
| `window_survey.csv`, `window_x*.csv` | `window_survey.py` |
| `window_sw.csv`, `window_ne.csv` | `window_robustness.py` |
| `recalib_cap.txt`, `recalib_doses.csv` | `recalibrate_cap.py` |
| `recalibration*.csv` | `recalibrate_cap.py`, `recal_layer.py` |
| `horizon_probe.csv` | `horizon_probe.py` |
| `dominance_probe.csv` | `dominance_probe.py` |
| `certificate.csv` | `cert_feasibility.py` |
| `exact_feasibility.csv` | `exact_pilot.py` |

CSV columns are self-describing: `solved`, `dose_to_arrival`, `viol_agents`, `planning_time`,
`cost`, `makespan_s`, `ct_nodes`, `low_expanded`, plus a per-file `reason`/`feasible` flag where
relevant.

---

## Notes on interpretation

* **Zero violations is structural.** The low level prunes labels that exceed the budget, so the
  planner cannot emit an over-exposed route; the experiments verify the implementation, they do
  not discover a property.
* **"No plan" is not "infeasible".** Instances where the deployed (bucketed) search returns nothing
  are reported as refusals. Bucketed dominance is incomplete, and the exact-dose pilots do not
  terminate with a complete search, so no exact reference exists for those instances.
* **Everything is offline.** Voxel grid, synthetic Gaussian plume, no flight dynamics, no
  controller, no communication model.

---

## Data source

Building geometry from **3D BAG** (TU Delft), © 3DBAG by the 3D Geoinformation group,
licensed CC BY 4.0 — <https://3dbag.nl/>. Only the derived voxel cache is built locally by
`bag_data.py`; no third-party data is redistributed here.

## Contact

For questions about the code, or for a pre-acceptance copy of the manuscript, please contact the
corresponding author of the paper.
