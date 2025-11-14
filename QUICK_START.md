
# КРАТКАЯ ИНСТРУКЦИЯ ПО ЗАПУСКУ

## Шаг 1: Установка
pip install pandas numpy matplotlib

## Шаг 2: Запуск бэктестинга
python main.py

## Шаг 3: Выбор режима
Выберите "1" для бэктестинга

## Шаг 4: Результаты
- backtest_results.png - визуализация
- trade_log.csv - детали сделок
- Консоль - полная статистика

## Настройка параметров
Откройте config.py и измените:
- STOP_LOSS_ATR_MULTIPLIER (по умолчанию 2.0)
- TAKE_PROFIT_ATR_MULTIPLIER (по умолчанию 3.5)
- COOLDOWN_MINUTES (по умолчанию 15)
- Временные ограничения торговли

## Использование своих данных
1. Подготовьте CSV с колонками: timestamp, open, high, low, close, volume
2. В main.py замените:
   data = DataLoader.load_from_csv('your_data.csv')

## Создание своей стратегии
1. Откройте strategies.py
2. Скопируйте LongPullbackStrategy как шаблон
3. Измените логику в on_candle()
4. Используйте в main.py
