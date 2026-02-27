"""
Canonical quality gate tier definitions for trace data.

See SCHEMA_CONTRACT.md for full documentation.
gpusim/v_agent_sim/src/workload/real_data_loader.py mirrors these values.
"""

VALIDATED_OK_RATE = 100.0
USABLE_STEP_OK_RATE = 95.0
USABLE_TASK_OK_RATE = 90.0

TIER_VALIDATED = "VALIDATED"
TIER_USABLE = "USABLE"
TIER_EXPLORATORY = "EXPLORATORY"

TIER_ORDER = {TIER_VALIDATED: 3, TIER_USABLE: 2, TIER_EXPLORATORY: 1}


def tier_at_least(tier: str, min_tier: str) -> bool:
    """Check whether *tier* meets or exceeds *min_tier*."""
    return TIER_ORDER.get(tier, 0) >= TIER_ORDER.get(min_tier, 0)
