"""
strategies_ml.py - Breakout стратегия с ML фильтром
Объединяет техническую стратегию пробоя с ML прогнозированием
"""

import pandas as pd
import numpy as np
from typing import Optional
import os

from trading_engine import TradingStrategy, Signal, OrderType, Trade, PositionSide
from market_context import MarketContext
from indicators import TechnicalIndicators
from config import config
from ml_predictor import MLPredictor


class MLEnhancedBreakoutStrategy(TradingStrategy):
    """
    Breakout стратегия с ML фильтром для ФЬЮЧЕРСОВ

    Логика:
    1. Определяем пробой диапазона (техническая часть)
    2. ML модель фильтрует ложные пробои
    3. Входим только если ML подтверждает направление

    Преимущества:
    - Меньше ложных входов
    - Выше Win Rate
    - ML учитывает 30+ факторов рынка
    """

    def __init__(self, ml_model_path: str = 'ml_model.pkl', 
                 use_ml: bool = True,
                 min_confidence: str = 'MEDIUM'):
        """
        Args:
            ml_model_path: Путь к обученной ML модели
            use_ml: Использовать ли ML фильтр
            min_confidence: Минимальная уверенность ML ('LOW', 'MEDIUM', 'HIGH')
        """
        self.lookback_candles = 15
        self.range_high = None
        self.range_low = None
        self.use_ml = use_ml
        self.min_confidence = min_confidence

        # ML предиктор
        self.ml_predictor = MLPredictor()

        # Загружаем модель если есть
        if use_ml and os.path.exists(ml_model_path):
            try:
                self.ml_predictor.load_model(ml_model_path)
                print(f"✅ ML модель загружена из {ml_model_path}")
            except Exception as e:
                print(f"⚠️ Не удалось загрузить ML модель: {e}")
                print("   Бот будет работать БЕЗ ML фильтра")
                self.use_ml = False
        else:
            if use_ml:
                print(f"⚠️ ML модель не найдена: {ml_model_path}")
                print("   Запустите: python train_ml_model.py")
                print("   Бот будет работать БЕЗ ML фильтра")
            self.use_ml = False

    def on_candle(self, df: pd.DataFrame, current_candle: pd.Series,
                  context: MarketContext, atr: float) -> Optional[Signal]:
        """
        Обработка новой свечи с ML фильтром

        Процесс:
        1. Техническая проверка пробоя
        2. ML прогноз направления
        3. Вход только если оба согласны
        """

        if atr == 0 or len(df) < self.lookback_candles + 10:
            return None

        current_price = current_candle['close']
        current_high = current_candle['high']
        current_low = current_candle['low']
        current_volume = current_candle['volume']

        # ======================================================================
        # ШАГИ 1-2: Определяем диапазон и проверяем фильтры
        # ======================================================================

        # Текущий диапазон (исключая текущую свечу)
        lookback_data = df.iloc[-(self.lookback_candles + 1):-1]
        self.range_high = lookback_data['high'].max()
        self.range_low = lookback_data['low'].min()
        range_size = self.range_high - self.range_low

        # Фильтр 1: Минимальный размер диапазона
        if range_size < atr * 2.0:
            return None

        # Фильтр 2: Повышенный объём на пробое
        avg_volume = lookback_data['volume'].mean()
        if current_volume < avg_volume * 1.2:
            return None

        # ======================================================================
        # ШАГ 3: ML ФИЛЬТР (если включён)
        # ======================================================================

        ml_allows_long = True
        ml_allows_short = True
        ml_info = ""

        if self.use_ml and self.ml_predictor.is_trained:
            ml_prediction = self.ml_predictor.predict(df)

            # Фильтруем по направлению и уверенности
            confidence_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
            min_conf_level = confidence_levels[self.min_confidence]
            current_conf_level = confidence_levels[ml_prediction['confidence']]

            if current_conf_level < min_conf_level:
                # Уверенность ML слишком низкая - не входим
                return None

            # Проверяем направление
            if ml_prediction['direction'] == 'UP':
                ml_allows_short = False  # ML говорит UP - запрещаем SHORT
            else:
                ml_allows_long = False   # ML говорит DOWN - запрещаем LONG

            ml_info = f"ML:{ml_prediction['direction']}({ml_prediction['probability']*100:.0f}%,{ml_prediction['confidence']})"

        # ======================================================================
        # ШАГ 4: ПРОВЕРКА ПРОБОЯ И ВХОД
        # ======================================================================

        # LONG: Пробой вверх
        if (current_high > self.range_high and 
            current_price > self.range_high and
            ml_allows_long):

            stop_loss = self.range_low - (atr * 0.5)
            risk = current_price - stop_loss
            take_profit = current_price + (risk * 3.0)  # R:R = 1:3

            reason = f"ml_breakout_long"
            if ml_info:
                reason += f"_{ml_info}"

            return Signal(
                action=OrderType.BUY,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason
            )

        # SHORT: Пробой вниз
        elif (current_low < self.range_low and 
              current_price < self.range_low and
              ml_allows_short):

            stop_loss = self.range_high + (atr * 0.5)
            risk = stop_loss - current_price
            take_profit = current_price - (risk * 3.0)  # R:R = 1:3

            reason = f"ml_breakout_short"
            if ml_info:
                reason += f"_{ml_info}"

            return Signal(
                action=OrderType.SELL,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason
            )

        return None

    def should_close_position(self, df: pd.DataFrame, current_price: float,
                             position: Trade) -> bool:
        """
        Закрываем позицию только по stop/take
        НЕ закрываем досрочно (для фьючерсов важно дать позиции развиться)
        """
        return False


class SimpleBreakoutStrategy(TradingStrategy):
    """
    Простая Breakout стратегия БЕЗ ML (для сравнения)
    """

    def __init__(self):
        self.lookback_candles = 15
        self.range_high = None
        self.range_low = None

    def on_candle(self, df: pd.DataFrame, current_candle: pd.Series,
                  context: MarketContext, atr: float) -> Optional[Signal]:
        """Простой пробой без ML"""

        if atr == 0 or len(df) < self.lookback_candles + 5:
            return None

        current_price = current_candle['close']
        current_high = current_candle['high']
        current_low = current_candle['low']

        # Диапазон
        lookback_data = df.iloc[-(self.lookback_candles + 1):-1]
        self.range_high = lookback_data['high'].max()
        self.range_low = lookback_data['low'].min()
        range_size = self.range_high - self.range_low

        # Фильтр размера
        if range_size < atr * 1.5:
            return None

        # LONG на пробое вверх
        if current_high > self.range_high and current_price > self.range_high:
            stop_loss = self.range_low - (atr * 0.5)
            risk = current_price - stop_loss
            take_profit = current_price + (risk * 2.5)

            return Signal(
                action=OrderType.BUY,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason="simple_breakout_long"
            )

        # SHORT на пробое вниз
        elif current_low < self.range_low and current_price < self.range_low:
            stop_loss = self.range_high + (atr * 0.5)
            risk = stop_loss - current_price
            take_profit = current_price - (risk * 2.5)

            return Signal(
                action=OrderType.SELL,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason="simple_breakout_short"
            )

        return None

    def should_close_position(self, df: pd.DataFrame, current_price: float,
                             position: Trade) -> bool:
        """Закрываем только по stop/take"""
        return False
