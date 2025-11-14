"""
config.py - Конфигурация для торговли ФЬЮЧЕРСАМИ (Si, RTS)
Оптимизировано для высоковолатильных инструментов
"""

from dataclasses import dataclass
from datetime import time

@dataclass
class FuturesConfig:
    """Параметры для торговли фьючерсами"""
    
    # Временные ограничения
    TRADING_START_TIME = time(10, 0)
    TRADING_END_TIME = time(18, 30)
    FORCE_CLOSE_TIME = time(23, 45)
    
    # ИСПРАВЛЕНО: Параметры волатильности для ФЬЮЧЕРСОВ
    ATR_PERIOD = 14
    VOLATILITY_LOOKBACK_DAYS = 3  # Меньше дней для фьючерсов (они быстрее)
    
    # КРИТИЧНО: Очень широкие стопы для фьючерсов!
    STOP_LOSS_ATR_MULTIPLIER = 6.0     # Было 4.0 → стало 6.0 ATR
    TAKE_PROFIT_ATR_MULTIPLIER = 9.0   # Было 6.5 → стало 9.0 ATR
    
    # Параметры откатов - более строгие для фьючерсов
    FIBONACCI_LEVELS = [0.5, 0.618]    # Только глубокие откаты!
    MIN_PULLBACK_LEVEL = 0.5           # Минимум 50% откат (было 0.382)
    
    # Индикаторы - более медленные для снижения шума
    EMA_FAST = 100                     # Было 50 → стало 100
    EMA_SLOW = 200                     # Осталось 200
    ADX_PERIOD = 14
    ADX_TREND_THRESHOLD = 30           # Было 25 → стало 30 (только сильные тренды!)
    ADX_NEUTRAL_THRESHOLD = 20
    
    # Более длинный кулдаун после стопа
    COOLDOWN_MINUTES = 30              # Было 20 → стало 30 минут
    COOLDOWN_ATR_MULTIPLIER = 1.5      # Было 1.0 → стало 1.5
    
    # Диапазонная торговля
    RANGE_DAYS_LOOKBACK = 3            # Меньше для фьючерсов
    RANGE_ENTRY_OFFSET_ATR = 0.5       # Было 0.4
    
    # Размер позиции
    POSITION_SIZE = 1
    
    # Тикер
    TICKER = "Si"  # Фьючерс на доллар
    
    # Бэктестинг
    INITIAL_CAPITAL = 100000
    COMMISSION = 0.0004

@dataclass
class StocksConfig:
    """Параметры для торговли"""
    
    # Временные ограничения
    TRADING_START_TIME = time(10, 0)
    TRADING_END_TIME = time(18, 30)
    FORCE_CLOSE_TIME = time(23, 45)
    
    # ИСПРАВЛЕНО: Параметры волатильности для ФЬЮЧЕРСОВ
    ATR_PERIOD = 14
    VOLATILITY_LOOKBACK_DAYS = 3  # Меньше дней для фьючерсов (они быстрее)
    
    # КРИТИЧНО: Очень широкие стопы для фьючерсов!
    STOP_LOSS_ATR_MULTIPLIER = 2.0     # Было 4.0 → стало 6.0 ATR
    TAKE_PROFIT_ATR_MULTIPLIER = 3.5   # Было 6.5 → стало 3.5 ATR
    
    # Параметры откатов - более строгие для фьючерсов
    FIBONACCI_LEVELS = [0.5, 0.618]    # Только глубокие откаты!
    MIN_PULLBACK_LEVEL = 0.5           # Минимум 50% откат (было 0.382)
    
    # Индикаторы - более медленные для снижения шума
    EMA_FAST = 50                     
    EMA_SLOW = 200                     # Осталось 200
    ADX_PERIOD = 14
    ADX_TREND_THRESHOLD = 25           
    ADX_NEUTRAL_THRESHOLD = 20
    
    # Более длинный кулдаун после стопа
    COOLDOWN_MINUTES = 30              # Было 20 → стало 30 минут
    COOLDOWN_ATR_MULTIPLIER = 1.5      # Было 1.0 → стало 1.5
    
    # Диапазонная торговля
    RANGE_DAYS_LOOKBACK = 3            # Меньше для фьючерсов
    RANGE_ENTRY_OFFSET_ATR = 0.5       # Было 0.4
    
    # Размер позиции
    POSITION_SIZE = 1
    
    # Тикер
    TICKER = "Si"  # Фьючерс на доллар
    
    # Бэктестинг
    INITIAL_CAPITAL = 100000
    COMMISSION = 0.0004

config = FuturesConfig()  # или StocksConfig()
