"""Confidence decay: periodic maintenance for fact freshness."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from loguru import logger

from ragnarbot.agent.memory.models import FactCategory

# Half-life in days per category
HALF_LIVES = {
    FactCategory.PERSONAL: 365,      # slow decay
    FactCategory.PROJECT: 90,        # projects change
    FactCategory.TECHNICAL: 180,     # tech evolves
    FactCategory.PREFERENCE: 120,    # preferences shift
    FactCategory.RELATIONSHIP: 365,  # relationships are stable
    FactCategory.EVENT: 30,          # events get stale fast
    FactCategory.OTHER: 90,
}

MIN_CONFIDENCE = 0.1  # don't decay below this


def compute_decayed_confidence(
    original_confidence: float,
    category: FactCategory,
    last_confirmed: datetime,
    now: datetime | None = None,
) -> float:
    """Compute current confidence after time decay.

    Uses exponential decay: c(t) = c0 * 2^(-t / half_life)
    """
    now = now or datetime.now()
    age_days = (now - last_confirmed).days
    if age_days <= 0:
        return original_confidence

    half_life = HALF_LIVES.get(category, 90)
    decayed = original_confidence * math.pow(2, -age_days / half_life)
    return max(decayed, MIN_CONFIDENCE)


class ConfidenceDecayService:
    """Periodic confidence decay for stored facts."""

    def __init__(self, backend: Any, batch_size: int = 100):
        self.backend = backend
        self.batch_size = batch_size

    async def run_decay(self) -> dict:
        """Apply confidence decay to all active facts.

        Returns summary of updates.
        """
        updated = 0
        invalidated = 0
        processed = 0

        try:
            facts = await self.backend.get_active_facts(limit=self.batch_size)
            now = datetime.now()

            for row in facts:
                processed += 1
                original = float(row.get("original_confidence", 1.0))
                category_str = row.get("category", "other")
                try:
                    category = FactCategory(category_str)
                except ValueError:
                    category = FactCategory.OTHER

                last_confirmed_raw = row.get("last_confirmed")
                if isinstance(last_confirmed_raw, str):
                    last_confirmed = datetime.fromisoformat(last_confirmed_raw)
                elif isinstance(last_confirmed_raw, datetime):
                    last_confirmed = last_confirmed_raw
                else:
                    continue

                new_conf = compute_decayed_confidence(original, category, last_confirmed, now)
                current_conf = float(row.get("confidence", 1.0))

                # Only update if changed significantly
                if abs(new_conf - current_conf) > 0.01:
                    fact_id = row["id"]
                    if new_conf <= MIN_CONFIDENCE:
                        await self.backend.invalidate_fact(fact_id)
                        invalidated += 1
                    else:
                        await self.backend.update_confidence(fact_id, new_conf)
                        updated += 1

        except Exception as e:
            logger.error(f"Confidence decay failed: {e}")

        result = {
            "processed": processed,
            "updated": updated,
            "invalidated": invalidated,
        }
        if updated or invalidated:
            logger.info(f"Confidence decay: {result}")
        return result
