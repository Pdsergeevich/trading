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


class MLEnhancedBreakoutStrategy(TradingStrategy):
    """
    Breakout стратегия + ML фильтр
    
    Требует предварительного обучения ML модели!
    Запустите: python train_ml_model.py
    """
    
    def __init__(self, ml_model_path: str = 'ml_model.pkl', 
                 use_ml: bool = True,
                 min_confidence: str = 'MEDIUM'):
        """
        Args:
            ml_model_path: Путь к обученной ML модели
            use_ml: Использовать ли ML фильтр
            min_confidence: Минимальная уверенность ('LOW', 'MEDIUM', 'HIGH')
        """
        self.lookback_candles = 15
        self.range_high = None
        self.range_low = None
        self.use_ml = use_ml
        self.min_confidence = min_confidence
        
        # ML предиктор
        self.ml_predictor = None
        
        if use_ml:
            try:
                from ml_predictor import MLPredictor
                self.ml_predictor = MLPredictor()
                
                # Загружаем модель если есть
                if os.path.exists(ml_model_path):
                    self.ml_predictor.load_model(ml_model_path)
                    print(f"✅ ML модель загружена из {ml_model_path}")
                else:
                    print(f"⚠️ ML модель не найдена: {ml_model_path}")
                    print("   Запустите: python train_ml_model.py")
                    print("   Бот будет работать БЕЗ ML фильтра")
                    self.use_ml = False
            except ImportError:
                print("⚠️ ml_predictor.py не найден!")
                print("   Бот будет работать БЕЗ ML фильтра")
                self.use_ml = False
    
    def on_candle(self, df: pd.DataFrame, current_candle: pd.Series,
                  context: MarketContext, atr: float) -> Optional[Signal]:
        """Обработка с ML фильтром"""
        
        if atr == 0 or len(df) < self.lookback_candles + 10:
            return None

        # ============================================================
        # ШАГ 1: ДАННЫЕ ТЕКУЩЕЙ СВЕЧИ
        # ============================================================
        current_price = current_candle['close']   # Цена закрытия
        current_high = current_candle['high']     # Максимум свечи
        current_low = current_candle['low']       # Минимум свечи
        current_volume = current_candle['volume'] # Объём

        # ============================================================
        # ШАГ 2: ОПРЕДЕЛЯЕМ ДИАПАЗОН
        # ============================================================
        lookback_data = df.iloc[-(self.lookback_candles + 1):-1]
        self.range_high = lookback_data['high'].max()
        self.range_low = lookback_data['low'].min()
        range_size = self.range_high - self.range_low

        # Фильтр размера диапазона
        if range_size < atr * 2.0:
            return None

        # Фильтр объёма
        avg_volume = lookback_data['volume'].mean()
        if current_volume < avg_volume * 1.2:
            return None

        # ============================================================
        # ШАГ 3: ML ФИЛЬТР (если включён)
        # ============================================================
        ml_allows_long = True
        ml_allows_short = True
        ml_info = ""
        
        if self.use_ml and self.ml_predictor and self.ml_predictor.is_trained:
            ml_prediction = self.ml_predictor.predict(df)
            
            # Проверяем уровень уверенности
            confidence_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
            min_conf_level = confidence_levels[self.min_confidence]
            current_conf_level = confidence_levels[ml_prediction['confidence']]
            
            if current_conf_level < min_conf_level:
                # Уверенность слишком низкая
                return None
            
            # Проверяем направление
            if ml_prediction['direction'] == 'UP':
                ml_allows_short = False  # Запрещаем SHORT
            else:
                ml_allows_long = False   # Запрещаем LONG
            
            ml_info = f"ML:{ml_prediction['direction']}({ml_prediction['probability']*100:.0f}%)"

        # ============================================================
        # ШАГ 4: LONG НА ПРОБОЕ ВВЕРХ
        # ============================================================
        if (current_high > self.range_high and 
            current_price > self.range_high and
            ml_allows_long):
            
            stop_loss = self.range_low - (atr * 0.5)
            risk = current_price - stop_loss
            take_profit = current_price + (risk * 3.0)

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

        # ============================================================
        # ШАГ 5: SHORT НА ПРОБОЕ ВНИЗ
        # ============================================================
        elif (current_low < self.range_low and 
              current_price < self.range_low and
              ml_allows_short):
            
            stop_loss = self.range_high + (atr * 0.5)
            risk = stop_loss - current_price
            take_profit = current_price - (risk * 3.0)

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
        """Закрываем только по stop/take"""
        return False
