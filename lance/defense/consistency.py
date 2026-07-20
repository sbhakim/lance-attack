"""C2: two-sided temporal-consistency screening.

Normal dynamics are fit on an early, presumed-clean prefix of the stream,
summarized by a global inter-event gap statistic ``g``. At runtime each edge's
node staleness (``t - last_update``) is compared to the band
``[band_low * g, band_high * g]``. Staleness above ``band_high * g`` marks a node
that was inactive and is then suddenly touched, the frozen-then-active signature
of memory attacks such as MemFreezing. Staleness below ``band_low * g`` marks an
abnormally dense burst, consistent with an injection flood. Edges outside the band
are down-weighted, which inverts a TDAP-style stealth bound: an attack that stays
hidden must remain inside the band.
"""
from __future__ import annotations

import numpy as np
import torch


class Consistency:
    def __init__(self, band_low: float = 0.25, band_high: float = 3.0,
                 clean_prefix_frac: float = 0.2, soft_weight: float = 0.7):
        self.band_low = band_low
        self.band_high = band_high
        self.clean_prefix_frac = clean_prefix_frac
        self.soft_weight = soft_weight
        self.g: float = 1.0      # fitted normal inter-event gap

    def fit(self, data) -> "Consistency":
        """Estimate the normal inter-event gap from the clean prefix."""
        src, dst, t, _ = data.split("train")
        cut = max(2, int(self.clean_prefix_frac * len(t)))
        last = {}
        gaps = []
        for u, v, tt in zip(src[:cut], dst[:cut], t[:cut]):
            for node in (u, v):
                if node in last:
                    gaps.append(tt - last[node])
                last[node] = tt
        self.g = float(np.median([gp for gp in gaps if gp > 0])) if gaps else 1.0
        if self.g <= 0:
            self.g = 1.0
        return self

    @torch.no_grad()
    def weights(self, model, batch) -> torch.Tensor:
        """Per-edge consistency weight in [soft_weight, 1] (two-sided band)."""
        stale_s = model.staleness(batch.src, batch.t)
        stale_d = model.staleness(batch.dst, batch.t)
        stale = torch.maximum(stale_s, stale_d)
        lo, hi = self.band_low * self.g, self.band_high * self.g
        inside = (stale >= lo) & (stale <= hi)
        w = torch.full_like(stale, self.soft_weight)
        w[inside] = 1.0
        return w
