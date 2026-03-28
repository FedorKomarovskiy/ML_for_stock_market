from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import DataConfig


INTERVAL_SECONDS: dict[str, int] = {
    "1min": 60,
    "3min": 180,
    "5min": 300,
    "15min": 900,
    "30min": 1_800,
    "1hour": 3_600,
    "2hour": 7_200,
    "4hour": 14_400,
    "6hour": 21_600,
    "8hour": 28_800,
    "12hour": 43_200,
    "1day": 86_400,
}


@dataclass(slots=True)
class KuCoinSpotClient:
    base_url: str = "https://api.kucoin.com"
    timeout_sec: int = 15
    session: requests.Session = field(init=False)

    def __post_init__(self) -> None:
        self.session = requests.Session()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        response = self.session.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "200000":
            raise RuntimeError(f"KuCoin request failed: {payload}")
        return payload["data"]

    def fetch_spot_candles(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        interval_sec = INTERVAL_SECONDS.get(interval)
        if interval_sec is None:
            raise ValueError(f"Unsupported interval: {interval}")
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        window_sec = interval_sec * 1_200
        cursor_start = start_ts
        rows: list[dict[str, float | int]] = []

        while cursor_start <= end_ts:
            cursor_end = min(end_ts, cursor_start + window_sec)
            data = self._get(
                "/api/v1/market/candles",
                {
                    "type": interval,
                    "symbol": symbol,
                    "startAt": cursor_start,
                    "endAt": cursor_end,
                },
            )
            if data:
                rows.extend(self._parse_spot_candle(raw) for raw in data)
            cursor_start = cursor_end + interval_sec

        if not rows:
            raise RuntimeError("No candles were returned by KuCoin.")

        frame = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["ts_sec"])
            .sort_values("ts_sec")
            .reset_index(drop=True)
        )
        frame["timestamp"] = pd.to_datetime(frame["ts_sec"], unit="s", utc=True)
        frame = frame[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ]
        ]
        return frame[(frame["timestamp"] >= start_dt) & (frame["timestamp"] <= end_dt)].reset_index(drop=True)

    @staticmethod
    def _parse_spot_candle(raw: list[Any]) -> dict[str, float | int]:
        return {
            "ts_sec": int(raw[0]),
            "open": float(raw[1]),
            "close": float(raw[2]),
            "high": float(raw[3]),
            "low": float(raw[4]),
            "volume": float(raw[5]),
            "turnover": float(raw[6]) if len(raw) > 6 else 0.0,
        }


def parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_or_download_data(cfg: DataConfig, force: bool = False) -> pd.DataFrame:
    cache_path = Path(cfg.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        frame = pd.read_csv(cache_path, parse_dates=["timestamp"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    client = KuCoinSpotClient()
    frame = client.fetch_spot_candles(
        symbol=cfg.symbol,
        interval=cfg.interval,
        start_dt=parse_utc(cfg.start_at),
        end_dt=parse_utc(cfg.end_at),
    )
    frame.to_csv(cache_path, index=False)
    return frame
