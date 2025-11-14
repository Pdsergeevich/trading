"""
strategies_futures.py - BREAKOUT стратегия для ФЬЮЧЕРСОВ
Оптимизирована для высоковолатильных инструментов (Si, RTS)
"""

import pandas as pd
import numpy as np
from typing import Optional
import os

from trading_engine import TradingStrategy, Signal, OrderType, Trade, PositionSide
from market_context import MarketContext
from indicators import TechnicalIndicators
from config import config


class BreakoutStrategy(TradingStrategy):
    """
    Простая Breakout стратегия для ФЬЮЧЕРСОВ
    
    Логика:
    1. Определяем диапазон последних N свечей (high/low)
    2. При пробое максимума → LONG
    3. При пробое минимума → SHORT
    4. Stop = за противоположной границей диапазона
    5. Take = 2.5x от риска
    """

    def __init__(self):
        self.lookback_candles = 15  # Смотрим на последние 15 свечей
        self.range_high = None
        self.range_low = None

    def on_candle(self, df: pd.DataFrame, current_candle: pd.Series,
                  context: MarketContext, atr: float) -> Optional[Signal]:
        """Обработка новой свечи для Breakout стратегии"""

        # Проверки
        if atr == 0 or len(df) < self.lookback_candles + 5:
            return None

        # ============================================================
        # ШАГ 1: ПОЛУЧАЕМ ДАННЫЕ ТЕКУЩЕЙ СВЕЧИ
        # ============================================================
        current_price = current_candle['close']  # Цена закрытия
        current_high = current_candle['high']    # Максимум свечи
        current_low = current_candle['low']      # Минимум свечи
        current_volume = current_candle['volume'] # Объём (опционально)

        # ============================================================
        # ШАГ 2: ОПРЕДЕЛЯЕМ ДИАПАЗОН (последние N свечей БЕЗ текущей)
        # ============================================================
        lookback_data = df.iloc[-(self.lookback_candles + 1):-1]
        self.range_high = lookback_data['high'].max()  # Максимум диапазона
        self.range_low = lookback_data['low'].min()    # Минимум диапазона
        range_size = self.range_high - self.range_low

        # ============================================================
        # ШАГ 3: ФИЛЬТР - Минимальный размер диапазона
        # ============================================================
        # Слишком узкий диапазон = шумовые движения
        min_range = atr * 1.5
        if range_size < min_range:
            return None  # Пропускаем

        # ============================================================
        # ШАГ 4: ФИЛЬТР - Повышенный объём (опционально)
        # ============================================================
        avg_volume = lookback_data['volume'].mean()
        if current_volume < avg_volume * 1.2:
            return None  # Низкий объём = ложный пробой

        # ============================================================
        # ШАГ 5: ПРОВЕРКА ПРОБОЯ ВВЕРХ → LONG
        # ============================================================
        if current_high > self.range_high:
            # Пробой максимума!
            
            # Подтверждение: цена закрылась ВЫШЕ максимума
            if current_price > self.range_high:
                
                # Стоп-лосс за минимумом диапазона
                stop_loss = self.range_low - (atr * 0.5)
                
                # Риск = расстояние от входа до стопа
                risk = current_price - stop_loss
                
                # Тейк-профит = 2.5x от риска (R:R = 1:2.5)
                take_profit = current_price + (risk * 2.5)

                return Signal(
                    action=OrderType.BUY,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason="breakout_long"
                )

        # ============================================================
        # ШАГ 6: ПРОВЕРКА ПРОБОЯ ВНИЗ → SHORT
        # ============================================================
        elif current_low < self.range_low:
            # Пробой минимума!
            
            # Подтверждение: цена закрылась НИЖЕ минимума
            if current_price < self.range_low:
                
                # Стоп-лосс за максимумом диапазона
                stop_loss = self.range_high + (atr * 0.5)
                
                # Риск = расстояние от входа до стопа
                risk = stop_loss - current_price
                
                # Тейк-профит = 2.5x от риска
                take_profit = current_price - (risk * 2.5)

                return Signal(
                    action=OrderType.SELL,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason="breakout_short"
                )

        # Нет пробоя - не входим
        return None

    def should_close_position(self, df: pd.DataFrame, current_price: float,
                             position: Trade) -> bool:
        """Закрываем только по stop/take, не досрочно"""
        return False
