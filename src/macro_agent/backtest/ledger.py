"""The trial ledger (spec 19.1).

Every strategy evaluation ever run appends one line. The count is permanent: it
is the denominator the deflated Sharpe uses, so the more variants are searched,
the higher the bar the winner has to clear. The file is only ever opened for
append.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class TrialLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(
        self,
        variant: str,
        params: dict,
        sharpe_per_period: float,
        sharpe_annual: float,
        data_range: tuple[str, str],
        holdout_used: bool,
    ) -> dict:
        blob = json.dumps(params, sort_keys=True, default=str)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "variant": variant,
            "params": json.loads(blob),
            "params_hash": hashlib.sha256(blob.encode()).hexdigest()[:12],
            "sharpe_per_period": sharpe_per_period,
            "sharpe_annual": sharpe_annual,
            "data_range": list(data_range),
            "holdout_used": holdout_used,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    @property
    def n_trials(self) -> int:
        return len(self.entries())

    def sharpes(self) -> list[float]:
        return [e["sharpe_per_period"] for e in self.entries()]
