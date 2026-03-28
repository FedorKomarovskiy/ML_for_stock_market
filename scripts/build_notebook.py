from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "btc_volatility_project.ipynb"


def build_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Прогнозирование волатильности BTC-USDT\n\n"
            "В ноутбуке собран полный воспроизводимый пайплайн проекта:\n"
            "- загрузка часовых данных KuCoin;\n"
            "- подготовка признаков;\n"
            "- baseline-модель GARCH(1,1);\n"
            "- DL-модель LSTM;\n"
            "- walk-forward валидация без утечки данных;\n"
            "- применение прогноза волатильности в vol-targeting стратегии;\n"
            "- расчёт Sharpe, Max Drawdown и Monte Carlo сценариев."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Импорт библиотек и подготовка путей\n"
            "Сначала подключим код проекта и посмотрим, где лежат артефакты полного прогона."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import display, Image\n\n"
            "ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n"
            "SRC = ROOT / 'src'\n"
            "if str(SRC) not in sys.path:\n"
            "    sys.path.insert(0, str(SRC))\n\n"
            "CONFIG_PATH = ROOT / 'config' / 'btc_kucoin_hourly.json'\n"
            "SUMMARY_PATH = ROOT / 'reports' / 'summary.json'\n"
            "PRED_PATH = ROOT / 'reports' / 'predictions.csv'\n"
            "FORECAST_METRICS_PATH = ROOT / 'reports' / 'forecast_metrics.csv'\n"
            "STRATEGY_METRICS_PATH = ROOT / 'reports' / 'strategy_metrics.csv'\n"
            "FEATURE_PATH = ROOT / 'data' / 'processed' / 'feature_frame.csv'\n"
            "FORECAST_PLOT = ROOT / 'reports' / 'forecast_comparison.png'\n"
            "EQUITY_PLOT = ROOT / 'reports' / 'equity_curve.png'\n"
            "MONTE_CARLO_PLOT = ROOT / 'reports' / 'monte_carlo_volatility.png'\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Полный прогон проекта\n"
            "Следующая ячейка запускает полный pipeline. Если артефакты уже построены, можно её не выполнять повторно."
        ),
        nbf.v4.new_code_cell(
            "from ml_for_stock_market.pipeline import run_pipeline\n\n"
            "# summary = run_pipeline(str(CONFIG_PATH))\n"
            "# print(json.dumps(summary, indent=2, ensure_ascii=False))"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Загрузка готовых результатов\n"
            "Подгрузим summary, метрики и подготовленный датасет."
        ),
        nbf.v4.new_code_cell(
            "summary = json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))\n"
            "predictions = pd.read_csv(PRED_PATH, parse_dates=['timestamp'])\n"
            "forecast_metrics = pd.read_csv(FORECAST_METRICS_PATH)\n"
            "strategy_metrics = pd.read_csv(STRATEGY_METRICS_PATH)\n"
            "feature_frame = pd.read_csv(FEATURE_PATH, parse_dates=['timestamp'])\n\n"
            "summary"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Проверка данных\n"
            "Посмотрим на объём выборки и на первые строки подготовленного признакового датасета."
        ),
        nbf.v4.new_code_cell(
            "print('Rows in feature frame:', len(feature_frame))\n"
            "print('Date range:', feature_frame['timestamp'].min(), '->', feature_frame['timestamp'].max())\n"
            "display(feature_frame.head())"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Качество прогноза волатильности\n"
            "Сравним baseline GARCH и LSTM по основным ошибкам прогнозирования."
        ),
        nbf.v4.new_code_cell("display(forecast_metrics)"),
        nbf.v4.new_markdown_cell(
            "## 6. Метрики применения модели\n"
            "Ниже показаны метрики для стратегии с управлением риском через прогноз волатильности."
        ),
        nbf.v4.new_code_cell("display(strategy_metrics)"),
        nbf.v4.new_markdown_cell(
            "## 7. Графики\n"
            "Покажем график фактической и прогнозной волатильности, а также кривую капитала стратегии."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(FORECAST_PLOT)))\n"
            "display(Image(filename=str(EQUITY_PLOT)))\n"
            "display(Image(filename=str(MONTE_CARLO_PLOT)))"
        ),
        nbf.v4.new_markdown_cell(
            "## 8. Краткие выводы\n"
            "1. LSTM лучше baseline GARCH по RMSE, MAE и корреляции с реализованной волатильностью.\n"
            "2. В прикладной части прогноз волатильности использовался для vol-targeting, а не для предсказания направления цены.\n"
            "3. Monte Carlo даёт диапазон возможной будущей волатильности и помогает в сценарном риск-анализе."
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")


if __name__ == "__main__":
    build_notebook()
