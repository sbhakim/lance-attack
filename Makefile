# DT-SHIELD developer workflow. Uses the `cti` conda env by default.
CONDA_RUN ?= conda run -n cti
DATA_ROOT ?= ../Dataset

.PHONY: help install lint test smoke prepare train-clean attack train-defended eval clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?# "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  # editable install + dev deps into the active env
	$(CONDA_RUN) pip install -e ".[dev]"

lint:  # static checks
	$(CONDA_RUN) ruff check lance scripts tests

test:  # unit tests
	$(CONDA_RUN) python -m pytest

smoke:  # fast end-to-end sanity run on a small data slice
	$(CONDA_RUN) python smoke_test.py --data-root $(DATA_ROOT)

prepare:  # cache processed tensors for a dataset (DATASET=mooc|wikipedia|bitcoinotc)
	$(CONDA_RUN) python scripts/prepare_data.py --config configs/$(DATASET).yaml

train-clean:  # train an undefended victim
	$(CONDA_RUN) python scripts/train.py --config configs/$(DATASET).yaml --defense none

train-defended:  # train with the DT-SHIELD defense
	$(CONDA_RUN) python scripts/train.py --config configs/$(DATASET).yaml --defense dtshield

attack:  # run HIA poisoning and cache the perturbed graph
	$(CONDA_RUN) python scripts/run_attack.py --config configs/$(DATASET).yaml

clean:
	rm -rf artifacts/*.pt artifacts/*.json __pycache__ .pytest_cache
