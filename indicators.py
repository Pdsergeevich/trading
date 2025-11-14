"""
indicators.py - Технические индикаторы для анализа рынка
Содержит функции для расчёта ATR, EMA, ADX и других индикаторов
"""

import pandas as pd
import numpy as np
from typing import Tuple

class TechnicalIndicators:
    """Класс для расчёта технических индикаторов"""

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Рассчитывает Average True Range (ATR) - среднюю волатильность

        Args:
            df: DataFrame с колонками high, low, close
            period: Период для расчёта ATR

        Returns:
            Series с значениями ATR
        """
        # True Range = максимум из трёх значений:
        # 1. High - Low
        # 2. abs(High - Previous Close)
        # 3. abs(Low - Previous Close)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR = экспоненциальное скользящее среднее от True Range
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """
        Рассчитывает экспоненциальную скользящую среднюю (EMA)

        Args:
            series: Серия данных (обычно цены закрытия)
            period: Период для расчёта EMA

        Returns:
            Series с значениями EMA
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Рассчитывает ADX (Average Directional Index) и направленные индикаторы
        ADX показывает силу тренда (не направление!)

        Args:
            df: DataFrame с колонками high, low, close
            period: Период для расчёта ADX

        Returns:
            Tuple (ADX, +DI, -DI)
        """
        # Расчёт +DM (положительное направленное движение) и -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        plus_dm = high_diff.copy()
        minus_dm = low_diff.copy()

        # +DM = High_diff, если High_diff > Low_diff и > 0, иначе 0
        plus_dm[((high_diff < low_diff) | (high_diff < 0))] = 0

        # -DM = Low_diff, если Low_diff > High_diff и > 0, иначе 0
        minus_dm[((low_diff < high_diff) | (low_diff < 0))] = 0

        # Сглаженные значения +DM и -DM
        plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()

        # ATR для нормализации
        atr = TechnicalIndicators.calculate_atr(df, period)

        # Направленные индикаторы (+DI и -DI)
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)

        # ADX = сглаженное значение DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx, plus_di, minus_di

    @staticmethod
    def calculate_fibonacci_levels(price_start: float, price_end: float, 
                                   is_uptrend: bool = True) -> dict:
        """
        Рассчитывает уровни Фибоначчи для определения точек входа на откатах

        Args:
            price_start: Начальная цена движения
            price_end: Конечная цена движения
            is_uptrend: True для восходящего тренда, False для нисходящего

        Returns:
            Словарь с уровнями Фибоначчи
        """
        diff = price_end - price_start

        levels = {
            'level_0': price_end,
            'level_23.6': price_end - diff * 0.236,
            'level_38.2': price_end - diff * 0.382,
            'level_50': price_end - diff * 0.5,
            'level_61.8': price_end - diff * 0.618,
            'level_78.6': price_end - diff * 0.786,
            'level_100': price_start
        }

        return levels

    @staticmethod
    def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        Находит экстремумы (swing high/low) для определения точек разворота

        Args:
            df: DataFrame с колонками high, low
            lookback: Количество свечей для поиска экстремумов

        Returns:
            Tuple (swing_highs, swing_lows)
        """
        swing_highs = df['high'].rolling(window=lookback*2+1, center=True).max()
        swing_lows = df['low'].rolling(window=lookback*2+1, center=True).min()

        # Экстремум = центральное значение в окне
        is_swing_high = df['high'] == swing_highs
        is_swing_low = df['low'] == swing_lows

        return is_swing_high, is_swing_low
