# TabPFN ranking experiment

`python MAIN.py` now defaults to `CONFIG["ranker_model"] = "tabpfn"`.
Install dependencies with `python -m pip install -r requirements-tabpfn.txt`.
The TabPFN dependency is pinned to 9.1.0, whose constructor was verified locally.
TabPFN downloads pretrained weights on first use; depending on the checkpoint,
Hugging Face authentication and acceptance of its model terms may be required.

The regressor predicts within-month return ranks in [-1, 1], rather than raw
returns or XGBoost decile labels. Scores feed the existing monthly portfolio
optimizer. Validation IC, validation-based CFI, the validation rank-to-return
calibration, train-plus-validation refitting, ablations, and shuffled-label
checks use the selected backend and preserve the existing temporal splits.

TabPFN uses a deterministic month-balanced sample of the eligible training
rows, capped at `tabpfn_max_train_samples` (1,000 on CPU/smoke, 10,000 on GPU).
Targets are ranked on the full monthly cross-section before sampling. This is
an approximation to fitting the full stock panel. Increase the cap only within
your checkpoint and hardware limits; the pretraining guardrails remain enabled.
`tabpfn_n_estimators` sets the ensemble size; `tabpfn_predict_batch` bounds query
batches. CFI requires many predictions and refits and can be expensive; set
`cfi_enabled = False` for a first smoke experiment.

Set `ranker_model = "xgb"` to run the existing XGBoost comparison, or `"ridge"`
for Ridge. After editing source/configuration, run `python build_main.py` to
regenerate MAIN.py. The older XGBoost descriptions in PIPELINE.md describe the
original baseline; this document describes the new default.

Synthetic adapter tests use a lightweight fake estimator and do not download
weights. A real-model smoke run still needs the competition data and checkpoint
access and is required before interpreting performance.
