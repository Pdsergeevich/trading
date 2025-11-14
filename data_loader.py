"""
data_loader.py - Модуль для загрузки и обработки исторических данных
Поддерживает загрузку из CSV и работу с Tinkoff API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import os

class DataLoader:
    """
    Класс для загрузки исторических данных свечей
    """

    @staticmethod
    def load_from_csv(file_path: str, date_column: str = 'timestamp') -> pd.DataFrame:
        """
        Загружает данные из CSV файла

        Args:
            file_path: Путь к CSV файлу
            date_column: Название колонки с датой/временем

        Returns:
            DataFrame с данными OHLCV
        """
        print(f"📂 Загрузка данных из {file_path}...")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        # Загружаем CSV
        df = pd.read_csv(file_path)

        # Проверяем наличие необходимых колонок
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(f"В CSV отсутствуют колонки: {missing}")

        # Преобразуем дату в индекс
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

        # Сортируем по времени
        df.sort_index(inplace=True)

        print(f"✅ Загружено {len(df)} свечей")
        print(f"Период: {df.index[0]} - {df.index[-1]}")

        return df

    @staticmethod
    def generate_sample_data(days: int = 10, interval_minutes: int = 1,
                            start_price: float = 100000.0) -> pd.DataFrame:
        """
        Генерирует синтетические данные для тестирования

        Args:
            days: Количество дней данных
            interval_minutes: Интервал свечей в минутах
            start_price: Начальная цена

        Returns:
            DataFrame с синтетическими OHLCV данными
        """
        print(f"🔄 Генерация синтетических данных за {days} дней...")

        # Временной диапазон (только торговые часы 10:00-18:45)
        start_date = datetime.now() - timedelta(days=days)
        timestamps = []

        for day in range(days):
            current_day = start_date + timedelta(days=day)
            # Торговые часы: 10:00 - 18:45
            day_start = current_day.replace(hour=10, minute=0, second=0, microsecond=0)
            day_end = current_day.replace(hour=18, minute=45, second=0, microsecond=0)

            current_time = day_start
            while current_time <= day_end:
                timestamps.append(current_time)
                current_time += timedelta(minutes=interval_minutes)

        # Генерируем цены (случайное блуждание с трендом)
        n_candles = len(timestamps)
        returns = np.random.normal(0.0002, 0.01, n_candles)  # Небольшой положительный тренд

        close_prices = [start_price]
        for ret in returns[1:]:
            close_prices.append(close_prices[-1] * (1 + ret))

        # Генерируем OHLC на основе close
        data = []
        for i, timestamp in enumerate(timestamps):
            close = close_prices[i]
            volatility = close * 0.002  # 0.2% волатильность

            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            open_price = np.random.uniform(low, high)

            # Корректируем close чтобы был между high и low
            close = np.clip(close, low, high)

            volume = np.random.randint(100, 10000)

            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        print(f"✅ Сгенерировано {len(df)} свечей")
        return df

    @staticmethod
    def save_to_csv(df: pd.DataFrame, file_path: str):
        """
        Сохраняет DataFrame в CSV файл

        Args:
            df: DataFrame для сохранения
            file_path: Путь для сохранения
        """
        df.to_csv(file_path)
        print(f"💾 Данные сохранены в {file_path}")

    @staticmethod
    def resample_data(df: pd.DataFrame, timeframe: str = '5min') -> pd.DataFrame:
        """
        Пересчитывает данные на другой таймфрейм

        Args:
            df: Исходный DataFrame
            timeframe: Новый таймфрейм ('5min', '15min', '1H' и т.д.)

        Returns:
            DataFrame с новым таймфреймом
        """
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        print(f"🔄 Данные пересчитаны на таймфрейм {timeframe}: {len(resampled)} свечей")
        return resampled

    @staticmethod
    def filter_trading_hours(df: pd.DataFrame, 
                            start_time: str = "10:00", 
                            end_time: str = "18:45") -> pd.DataFrame:
        """
        Фильтрует данные, оставляя только торговые часы

        Args:
            df: Исходный DataFrame
            start_time: Время начала торговли (HH:MM)
            end_time: Время окончания торговли (HH:MM)

        Returns:
            Отфильтрованный DataFrame
        """
        from datetime import time

        start_h, start_m = map(int, start_time.split(':'))
        end_h, end_m = map(int, end_time.split(':'))

        start = time(start_h, start_m)
        end = time(end_h, end_m)

        filtered = df[(df.index.time >= start) & (df.index.time <= end)]

        print(f"🕐 Отфильтровано по торговым часам ({start_time}-{end_time}): {len(filtered)} свечей")
        return filtered


# Функции-помощники для работы с Tinkoff Invest API
class TinkoffDataLoader:
    """
    Загрузчик данных через Tinkoff Invest API
    (Требует установки tinkoff-investments и токена)
    """

    def __init__(self, token: str):
        """
        Args:
            token: Токен Tinkoff Invest API
        """
        try:
            from tinkoff.invest import Client, CandleInterval
            from tinkoff.invest.utils import now
            self.token = token
            self.Client = Client
            self.CandleInterval = CandleInterval
            self.now = now
        except ImportError:
            raise ImportError(
                "Для работы с Tinkoff API установите: pip install tinkoff-investments"
            )

    def load_candles(self, figi: str, days: int = 10, 
                    interval: str = '1min') -> pd.DataFrame:
        """
        Загружает исторические свечи через Tinkoff API

        Args:
            figi: FIGI инструмента
            days: Количество дней истории
            interval: Интервал ('1min', '5min', '1hour', '1day')

        Returns:
            DataFrame с данными
        """
        interval_map = {
            '1min': self.CandleInterval.CANDLE_INTERVAL_1_MIN,
            '5min': self.CandleInterval.CANDLE_INTERVAL_5_MIN,
            '15min': self.CandleInterval.CANDLE_INTERVAL_15_MIN,
            '1hour': self.CandleInterval.CANDLE_INTERVAL_HOUR,
            '1day': self.CandleInterval.CANDLE_INTERVAL_DAY
        }

        with self.Client(self.token) as client:
            from datetime import timedelta

            end = self.now()
            start = end - timedelta(days=days)

            candles = client.market_data.get_candles(
                figi=figi,
                from_=start,
                to=end,
                interval=interval_map.get(interval, self.CandleInterval.CANDLE_INTERVAL_1_MIN)
            )

            data = []
            for candle in candles.candles:
                data.append({
                    'timestamp': candle.time,
                    'open': self._quotation_to_float(candle.open),
                    'high': self._quotation_to_float(candle.high),
                    'low': self._quotation_to_float(candle.low),
                    'close': self._quotation_to_float(candle.close),
                    'volume': candle.volume
                })

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)

            print(f"✅ Загружено {len(df)} свечей из Tinkoff API")
            return df

    @staticmethod
    def _quotation_to_float(quotation) -> float:
        """Конвертирует Quotation в float"""
        return quotation.units + quotation.nano / 1e9
