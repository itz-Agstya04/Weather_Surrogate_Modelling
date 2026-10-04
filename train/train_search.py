import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ablations.search_proposed import run


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("budget_hours", nargs="?", type=float, default=1.0)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if args.dry:
        run(budget_hours=args.budget_hours, top_k=1, screen_epochs=1, full_epochs=1, rollout_epochs=0)
    else:
        run(budget_hours=args.budget_hours, top_k=1, screen_epochs=5, full_epochs=15, rollout_epochs=1)
