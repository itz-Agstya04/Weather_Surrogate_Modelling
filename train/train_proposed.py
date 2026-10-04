"""Compatibility entry point for budget-aware proposed-model experiments."""

from ablations.search_proposed import run


if __name__ == "__main__":
    run(budget_hours=1.0, top_k=1, screen_epochs=5, full_epochs=15, rollout_epochs=1)
