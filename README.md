# BTC-USDT Volatility Forecasting Project

Репозиторий содержит полностью воспроизводимый проект по прогнозированию волатильности для `BTC-USDT` на часовых данных `KuCoin`.

В проекте реализованы:
- загрузка исторических OHLCV-данных через публичный KuCoin API;
- формирование признаков по часовым свечам;
- baseline-модель `GARCH(1,1)`;
- deep learning модель `LSTM`;
- `walk-forward` валидация без утечки данных;
- применение прогноза волатильности в стратегии `vol-targeting`;
- расчёт `Sharpe Ratio`, `Max Drawdown`, `CAGR`;
- Monte Carlo моделирование будущей волатильности;
- Jupyter Notebook с комментариями и визуализациями.

## 1. Идея проекта

Вместо попытки предсказывать направление цены BTC, проект решает более прикладную задачу: прогнозирует **будущую реализованную волатильность** на горизонте `24 часа`.

Дальше этот прогноз используется в риск-менеджменте:
- если модель ожидает низкую волатильность, стратегия держит большую долю актива;
- если модель ожидает высокую волатильность, стратегия автоматически снижает экспозицию.

Такой подход хорошо подходит для сравнения `GARCH` и `LSTM`, потому что обе модели работают именно с волатильностью.

## 2. Выбор актива и данных

- Актив: `BTC-USDT`
- Источник: публичный `KuCoin API`
- Частота: `1 hour`
- Период: с `2019-01-01` по `2026-03-01`

Почему выбран именно `BTC-USDT`:
- высокая ликвидность;
- длинная история;
- удобный публичный API без обязательных ключей для research-части.

## 3. Структура проекта

- `config/btc_kucoin_hourly.json` - основной конфиг проекта
- `src/ml_for_stock_market/data.py` - загрузка данных с KuCoin
- `src/ml_for_stock_market/features.py` - признаки и целевая переменная
- `src/ml_for_stock_market/garch.py` - baseline GARCH
- `src/ml_for_stock_market/deep_model.py` - LSTM для прогноза волатильности
- `src/ml_for_stock_market/walkforward.py` - walk-forward без ликов данных
- `src/ml_for_stock_market/strategy.py` - vol-targeting стратегия и метрики
- `src/ml_for_stock_market/monte_carlo.py` - Monte Carlo сценарии по волатильности
- `src/ml_for_stock_market/pipeline.py` - единый исследовательский пайплайн
- `run_project.py` - точка входа
- `notebooks/btc_volatility_project.ipynb` - notebook для сдачи
- `tests/` - unit-тесты

## 4. Установка

```powershell
cd "C:\Users\admin\ML_for_stock_market"
powershell -ExecutionPolicy Bypass -File .\scripts\project.ps1 -Action install
```

## 5. Запуск проекта

Полный прогон:

```powershell
cd "C:\Users\admin\ML_for_stock_market"
powershell -ExecutionPolicy Bypass -File .\scripts\project.ps1 -Action run
```

Результаты сохраняются в:
- `reports/forecast_metrics.csv`
- `reports/strategy_metrics.csv`
- `reports/summary.json`
- `reports/forecast_comparison.png`
- `reports/equity_curve.png`
- `reports/monte_carlo_volatility.png`

## 6. Тесты

```powershell
cd "C:\Users\admin\ML_for_stock_market"
powershell -ExecutionPolicy Bypass -File .\scripts\project.ps1 -Action test
```

## 7. Notebook

Собрать ноутбук:

```powershell
cd "C:\Users\admin\ML_for_stock_market"
.\.venv\Scripts\python.exe .\scripts\build_notebook.py
```

Открыть ноутбук:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\project.ps1 -Action notebook
```

## 8. Что внутри модели

### Baseline

`GARCH(1,1)` используется как классический референс для прогнозирования условной волатильности.

### Deep Learning

`LSTM` получает последовательность из `48` последних часов и предсказывает годовую реализованную волатильность на следующие `24 часа`.

### Признаки

В модели используются:
- лог-доходности;
- абсолютные доходности;
- внутридневной ценовой диапазон;
- z-score объёма;
- momentum на `6h` и `24h`;
- rolling volatility на `6h`, `24h`, `72h`, `168h`;
- отношения краткосрочной и долгосрочной волатильности;
- EMA gap;
- циклические временные признаки по часу и дню недели.

## 9. Walk-Forward схема

Используется расширяющееся окно:
- начальное обучение: `365 дней`
- test fold: `90 дней`
- шаг сдвига: `90 дней`

На каждом fold:
1. обе модели обучаются только на прошлом;
2. прогнозируют волатильность на следующем интервале;
3. результаты собираются в единый out-of-sample набор.

## 10. Прикладная часть

Прогноз волатильности используется в `vol-targeting` стратегии:

```text
exposure_t = min(max_leverage, target_vol / predicted_vol_t)
```

Это позволяет:
- снижать позицию в периоды ожидаемого роста риска;
- поддерживать более стабильный риск-профиль;
- сравнить, насколько полезен прогноз волатильности в реальном применении.

## 11. Что нужно для полного тестирования

Для research-части от пользователя ничего дополнительного не требуется, кроме:
- доступа в интернет для первичной загрузки данных KuCoin;
- установленного Python 3.12;
- возможности установить зависимости из `requirements.txt`.

Что **не нужно** для этого проекта:
- API-ключ KuCoin;
- торговый счёт;
- доступ к T-Invest.

Что понадобится отдельно, если нужно будет:
- запушить репозиторий в GitHub: авторизация GitHub на этой машине;
- поменять актив: от вас нужен только новый тикер и подтверждение, что оставляем KuCoin как источник.

## 12. Быстрая проверка перед сдачей

```powershell
cd "C:\Users\admin\ML_for_stock_market"
powershell -ExecutionPolicy Bypass -File .\scripts\project.ps1 -Action test
powershell -ExecutionPolicy Bypass -File .\scripts\project.ps1 -Action run
.\.venv\Scripts\python.exe .\scripts\build_notebook.py
```
