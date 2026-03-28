from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_src_on_path(repo_root: Path) -> None:
    src_path = str((repo_root / "src").resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BTC-USDT volatility forecasting project.")
    parser.add_argument(
        "--config",
        default="config/btc_kucoin_hourly.json",
        help="Path to JSON config.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore local cached market data and re-download candles.",
    )
    return parser.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    _ensure_src_on_path(repo_root)

    from ml_for_stock_market.pipeline import run_pipeline

    args = parse_args()
    summary = run_pipeline(
        config_path=str((repo_root / args.config).resolve()),
        force_download=args.force_download,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
