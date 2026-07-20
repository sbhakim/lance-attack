"""CLI: sanity-check / summarize a dataset referenced by a config.

Example:
    python scripts/prepare_data.py --config configs/mooc.yaml
"""
from __future__ import annotations

import argparse

try:
    from scripts._common import Config, load_data
except ModuleNotFoundError:
    from _common import Config, load_data
from lance.utils import get_logger

_LOG = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    data = load_data(cfg)
    _LOG.info(
        f"{cfg.data.name}: nodes={data.num_nodes} feats={data.num_feats} "
        f"events={len(data.src)} | train/val/test="
        f"{data.num_train_events()}/{len(data.val_idx)}/{len(data.test_idx)} "
        f"| t=[{data.t.min():.1f},{data.t.max():.1f}]"
    )


if __name__ == "__main__":
    main()
