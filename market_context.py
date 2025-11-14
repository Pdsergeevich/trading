"""
market_context.py - Модуль для определения контекста рынка
Определяет: положительный (лонг), отрицательный (шорт) или нейтральный контекст
ИСПРАВЛЕНО: добавлен фильтр частой смены контекста
"""

import pandas as pd
from enum import Enum
from indicators import TechnicalIndicators
from config import config

class MarketContext(Enum):
    """Возможные состояния рынка"""
    BULLISH = "bullish"      # Положительный контекст - торгуем в лонг
    BEARISH = "bearish"      # Отрицательный контекст - торгуем в шорт
    NEUTRAL = "neutral"      # Нейтральный контекст - диапазонная торговля
    UNKNOWN = "unknown"      # Недостаточно данных

class MarketContextAnalyzer:
    """Класс для анализа и определения контекста рынка"""
    
    def __init__(self):
        self.current_context = MarketContext.UNKNOWN
        self.ema_fast = None
        self.ema_slow = None
        self.adx = None
        self.plus_di = None
        self.minus_di = None
        
        # ИСПРАВЛЕНИЕ: Добавляем счётчик подтверждений для фильтрации
        self.context_confirmation_counter = 0
        self.pending_context = None
        self.CONFIRMATION_CANDLES = 3  # Нужно 5 свечей подтверждения для фьючерсов: 3 свечи (быстрее)
    
    def update_context(self, df: pd.DataFrame) -> MarketContext:
        """
        Обновляет и возвращает текущий контекст рынка
        ИСПРАВЛЕНО: Теперь требуется подтверждение в течение 5 свечей
        
        Логика определения контекста:
        1. BULLISH (восходящий тренд):
           - Цена выше EMA(50) И EMA(200)
           - ADX > 25 (сильный тренд)
           - +DI > -DI (восходящее направление)
        
        2. BEARISH (нисходящий тренд):
           - Цена ниже EMA(50) И EMA(200)
           - ADX > 25 (сильный тренд)
           - -DI > +DI (нисходящее направление)
        
        3. NEUTRAL (боковик):
           - ADX < 20 (слабый тренд)
           - Рынок движется в диапазоне
        
        Args:
            df: DataFrame с историческими данными
            
        Returns:
            Текущий контекст рынка
        """
        if len(df) < max(config.EMA_SLOW, config.ADX_PERIOD) + 10:
            return MarketContext.UNKNOWN
        
        # Рассчитываем индикаторы
        self.ema_fast = TechnicalIndicators.calculate_ema(df['close'], config.EMA_FAST)
        self.ema_slow = TechnicalIndicators.calculate_ema(df['close'], config.EMA_SLOW)
        self.adx, self.plus_di, self.minus_di = TechnicalIndicators.calculate_adx(
            df, config.ADX_PERIOD
        )
        
        # Берём последние значения
        current_price = df['close'].iloc[-1]
        current_ema_fast = self.ema_fast.iloc[-1]
        current_ema_slow = self.ema_slow.iloc[-1]
        current_adx = self.adx.iloc[-1]
        current_plus_di = self.plus_di.iloc[-1]
        current_minus_di = self.minus_di.iloc[-1]
        
        # Проверяем на NaN
        if pd.isna(current_adx) or pd.isna(current_ema_fast) or pd.isna(current_ema_slow):
            return MarketContext.UNKNOWN
        
        # ОПРЕДЕЛЯЕМ НОВЫЙ КОНТЕКСТ (предварительно)
        new_context = None
        
        # 1. Нейтральный рынок (слабый тренд)
        if current_adx < config.ADX_NEUTRAL_THRESHOLD:
            new_context = MarketContext.NEUTRAL
        
        # 2. Сильный тренд (ADX >= 25)
        elif current_adx >= config.ADX_TREND_THRESHOLD:
            # Восходящий тренд
            if (current_price > current_ema_fast and 
                current_price > current_ema_slow and
                current_plus_di > current_minus_di):
                new_context = MarketContext.BULLISH
            
            # Нисходящий тренд
            elif (current_price < current_ema_fast and 
                  current_price < current_ema_slow and
                  current_minus_di > current_plus_di):
                new_context = MarketContext.BEARISH
            else:
                new_context = MarketContext.NEUTRAL
        else:
            new_context = MarketContext.NEUTRAL
        
        # ИСПРАВЛЕНИЕ: Фильтр частой смены контекста
        # Требуем подтверждение нового контекста в течение N свечей
        if new_context != self.current_context:
            # Новый контекст отличается от текущего
            if self.pending_context == new_context:
                # Продолжаем подтверждение того же нового контекста
                self.context_confirmation_counter += 1
                
                if self.context_confirmation_counter >= self.CONFIRMATION_CANDLES:
                    # Достаточно подтверждений - меняем контекст
                    self.current_context = new_context
                    self.context_confirmation_counter = 0
                    self.pending_context = None
            else:
                # Начинаем подтверждение нового контекста
                self.pending_context = new_context
                self.context_confirmation_counter = 1
        else:
            # Контекст не изменился - сбрасываем счётчик
            self.context_confirmation_counter = 0
            self.pending_context = None
        
        return self.current_context
    
    def get_context_info(self) -> dict:
        """
        Возвращает детальную информацию о текущем контексте
        
        Returns:
            Словарь с информацией об индикаторах
        """
        return {
            'context': self.current_context.value if self.current_context else 'unknown',
            'adx': self.adx.iloc[-1] if self.adx is not None and len(self.adx) > 0 else None,
            'plus_di': self.plus_di.iloc[-1] if self.plus_di is not None and len(self.plus_di) > 0 else None,
            'minus_di': self.minus_di.iloc[-1] if self.minus_di is not None and len(self.minus_di) > 0 else None,
            'ema_fast': self.ema_fast.iloc[-1] if self.ema_fast is not None and len(self.ema_fast) > 0 else None,
            'ema_slow': self.ema_slow.iloc[-1] if self.ema_slow is not None and len(self.ema_slow) > 0 else None,
            'pending_context': self.pending_context.value if self.pending_context else None,
            'confirmations': self.context_confirmation_counter
        }
