"""
strategies.py - Стратегии для ФЬЮЧЕРСОВ (Si, RTS)
ОПТИМИЗИРОВАНО для высоковолатильных инструментов
"""

import pandas as pd
import numpy as np
from typing import Optional

from trading_engine import TradingStrategy, Signal, OrderType, Trade, PositionSide
from market_context import MarketContext
from indicators import TechnicalIndicators
from config import config


class LongPullbackStrategy(TradingStrategy):
    """
    Лонг-стратегия для ФЬЮЧЕРСОВ
    ДОБАВЛЕНО: Фильтр минимального размера движения
    """

    def __init__(self):
        self.last_swing_high = None
        self.last_swing_low = None
        self.looking_for_entry = False

    def on_candle(self, df: pd.DataFrame, current_candle: pd.Series,
                  context: MarketContext, atr: float) -> Optional[Signal]:
        """Обработка новой свечи для лонг-стратегии"""

        # Работаем только в бычьем контексте
        if context != MarketContext.BULLISH:
            self.looking_for_entry = False
            return None

        if atr == 0 or len(df) < 20:
            return None

        current_price = current_candle['close']

        # Ищем последние swing high/low
        swing_highs, swing_lows = TechnicalIndicators.find_swing_points(df, lookback=5)

        # Обновляем последний максимум
        if swing_highs.iloc[-6:-1].any():
            idx = swing_highs.iloc[-6:-1].idxmax() if swing_highs.iloc[-6:-1].any() else None
            if idx is not None:
                self.last_swing_high = df.loc[idx, 'high']

        # Обновляем последний минимум  
        if swing_lows.iloc[-6:-1].any():
            idx = swing_lows.iloc[-6:-1].idxmin() if swing_lows.iloc[-6:-1].any() else None
            if idx is not None:
                self.last_swing_low = df.loc[idx, 'low']

        # Проверяем, есть ли откат
        if self.last_swing_high is not None and self.last_swing_low is not None:
            if self.last_swing_high > self.last_swing_low:  # Восходящее движение

                # ✅ НОВЫЙ ФИЛЬТР: Минимальный размер движения для фьючерсов
                swing_range = self.last_swing_high - self.last_swing_low
                min_move = atr * 2.5  # Минимум 2.5 ATR движения
                
                if swing_range < min_move:
                    # Слишком маленькое движение - пропускаем
                    return None

                # Рассчитываем уровни Фибоначчи
                fib_levels = TechnicalIndicators.calculate_fibonacci_levels(
                    self.last_swing_low, self.last_swing_high, is_uptrend=True
                )

                # ✅ ИЗМЕНЕНО: Проверяем только уровни 50% и 61.8% (глубокие откаты)
                fib_50 = fib_levels['level_50']
                fib_618 = fib_levels['level_61.8']

                # Откат к уровню 61.8% (более глубокий откат)
                if current_price <= fib_618 and current_price >= self.last_swing_low:
                    # Дополнительная проверка: цена должна снова пойти вверх (подтверждение)
                    if len(df) >= 2 and current_candle['close'] > df.iloc[-2]['close']:

                        # Генерируем сигнал на вход
                        stop_loss = current_price - (atr * config.STOP_LOSS_ATR_MULTIPLIER)
                        take_profit = current_price + (atr * config.TAKE_PROFIT_ATR_MULTIPLIER)

                        return Signal(
                            action=OrderType.BUY,
                            price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            reason=f"long_pullback_fib61.8"
                        )

                # Откат к уровню 50%
                elif current_price <= fib_50 and current_price >= fib_618:
                    if len(df) >= 2 and current_candle['close'] > df.iloc[-2]['close']:

                        stop_loss = current_price - (atr * config.STOP_LOSS_ATR_MULTIPLIER)
                        take_profit = current_price + (atr * config.TAKE_PROFIT_ATR_MULTIPLIER)

                        return Signal(
                            action=OrderType.BUY,
                            price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            reason=f"long_pullback_fib50"
                        )

        return None

    def should_close_position(self, df: pd.DataFrame, current_price: float,
                             position: Trade) -> bool:
        """Проверка, нужно ли закрыть лонг позицию"""
        return False


class ShortPullbackStrategy(TradingStrategy):
    """
    Шорт-стратегия для ФЬЮЧЕРСОВ
    ДОБАВЛЕНО: Фильтр минимального размера движения
    """

    def __init__(self):
        self.last_swing_high = None
        self.last_swing_low = None

    def on_candle(self, df: pd.DataFrame, current_candle: pd.Series,
                  context: MarketContext, atr: float) -> Optional[Signal]:
        """Обработка новой свечи для шорт-стратегии"""

        # Работаем только в медвежьем контексте
        if context != MarketContext.BEARISH:
            return None

        if atr == 0 or len(df) < 20:
            return None

        current_price = current_candle['close']

        # Ищем последние swing high/low
        swing_highs, swing_lows = TechnicalIndicators.find_swing_points(df, lookback=5)

        # Обновляем последний максимум
        if swing_highs.iloc[-6:-1].any():
            idx = swing_highs.iloc[-6:-1].idxmax() if swing_highs.iloc[-6:-1].any() else None
            if idx is not None:
                self.last_swing_high = df.loc[idx, 'high']

        # Обновляем последний минимум
        if swing_lows.iloc[-6:-1].any():
            idx = swing_lows.iloc[-6:-1].idxmin() if swing_lows.iloc[-6:-1].any() else None
            if idx is not None:
                self.last_swing_low = df.loc[idx, 'low']

        # Проверяем, есть ли откат вверх
        if self.last_swing_high is not None and self.last_swing_low is not None:
            if self.last_swing_low < self.last_swing_high:  # Нисходящее движение

                # ✅ НОВЫЙ ФИЛЬТР: Минимальный размер движения для фьючерсов
                swing_range = self.last_swing_high - self.last_swing_low
                min_move = atr * 2.5  # Минимум 2.5 ATR движения
                
                if swing_range < min_move:
                    # Слишком маленькое движение - пропускаем
                    return None

                # Рассчитываем уровни Фибоначчи для нисходящего движения
                fib_levels = TechnicalIndicators.calculate_fibonacci_levels(
                    self.last_swing_high, self.last_swing_low, is_uptrend=False
                )

                # Для шорта - откат вверх
                movement_range = self.last_swing_high - self.last_swing_low
                fib_50_level = self.last_swing_low + movement_range * 0.5
                fib_618_level = self.last_swing_low + movement_range * 0.618

                # Откат к уровню 61.8%
                if current_price >= fib_618_level and current_price <= self.last_swing_high:
                    # Подтверждение: цена снова идёт вниз
                    if len(df) >= 2 and current_candle['close'] < df.iloc[-2]['close']:

                        stop_loss = current_price + (atr * config.STOP_LOSS_ATR_MULTIPLIER)
                        take_profit = current_price - (atr * config.TAKE_PROFIT_ATR_MULTIPLIER)

                        return Signal(
                            action=OrderType.SELL,
                            price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            reason=f"short_pullback_fib61.8"
                        )

                # Откат к уровню 50%
                elif current_price >= fib_50_level and current_price <= fib_618_level:
                    if len(df) >= 2 and current_candle['close'] < df.iloc[-2]['close']:

                        stop_loss = current_price + (atr * config.STOP_LOSS_ATR_MULTIPLIER)
                        take_profit = current_price - (atr * config.TAKE_PROFIT_ATR_MULTIPLIER)

                        return Signal(
                            action=OrderType.SELL,
                            price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            reason=f"short_pullback_fib50"
                        )

        return None

    def should_close_position(self, df: pd.DataFrame, current_price: float,
                             position: Trade) -> bool:
        """Проверка, нужно ли закрыть шорт позицию"""
        return False


class NeutralRangeStrategy(TradingStrategy):
    """
    Нейтральная стратегия: Диапазонная торговля
    БЕЗ ИЗМЕНЕНИЙ для фьючерсов
    """

    def __init__(self):
        self.range_high = None
        self.range_low = None
        self.last_update_day = None

    def _update_range(self, df: pd.DataFrame):
        """Обновляет диапазон торговли на основе последних дней"""
        if len(df) < config.RANGE_DAYS_LOOKBACK * 390:
            return

        lookback_candles = config.RANGE_DAYS_LOOKBACK * 390
        recent_data = df.iloc[-lookback_candles:]

        self.range_high = recent_data['high'].max()
        self.range_low = recent_data['low'].min()

    def on_candle(self, df: pd.DataFrame, current_candle: pd.Series,
                  context: MarketContext, atr: float) -> Optional[Signal]:
        """Обработка новой свечи для нейтральной стратегии"""

        if context != MarketContext.NEUTRAL:
            return None

        if atr == 0:
            return None

        self._update_range(df)

        if self.range_high is None or self.range_low is None:
            return None

        current_price = current_candle['close']

        offset = atr * config.RANGE_ENTRY_OFFSET_ATR
        buy_level = self.range_low + offset
        sell_level = self.range_high - offset

        # Сигнал на покупку около минимума
        if current_price <= buy_level:
            if len(df) >= 2 and current_candle['close'] > df.iloc[-2]['low']:

                stop_loss = self.range_low - (atr * 0.5)
                take_profit = sell_level

                return Signal(
                    action=OrderType.BUY,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason="neutral_range_buy_low"
                )

        # Сигнал на продажу около максимума
        elif current_price >= sell_level:
            if len(df) >= 2 and current_candle['close'] < df.iloc[-2]['high']:

                stop_loss = self.range_high + (atr * 0.5)
                take_profit = buy_level

                return Signal(
                    action=OrderType.SELL,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason="neutral_range_sell_high"
                )

        return None

    def should_close_position(self, df: pd.DataFrame, current_price: float,
                             position: Trade) -> bool:
        """Проверка, нужно ли закрыть позицию в нейтральной стратегии"""
        if self.range_high is None or self.range_low is None:
            return False

        if position.side == PositionSide.LONG:
            if current_price < self.range_low * 0.995:
                return True

        if position.side == PositionSide.SHORT:
            if current_price > self.range_high * 1.005:
                return True

        return False


class CombinedStrategy(TradingStrategy):
    """
    Комбинированная стратегия для ФЬЮЧЕРСОВ
    Использует все три стратегии в зависимости от контекста
    """

    def __init__(self):
        self.long_strategy = LongPullbackStrategy()
        self.short_strategy = ShortPullbackStrategy()
        self.neutral_strategy = NeutralRangeStrategy()

    def on_candle(self, df: pd.DataFrame, current_candle: pd.Series,
                  context: MarketContext, atr: float) -> Optional[Signal]:
        """Выбирает стратегию в зависимости от контекста"""

        if context == MarketContext.BULLISH:
            return self.long_strategy.on_candle(df, current_candle, context, atr)

        elif context == MarketContext.BEARISH:
            return self.short_strategy.on_candle(df, current_candle, context, atr)

        elif context == MarketContext.NEUTRAL:
            return self.neutral_strategy.on_candle(df, current_candle, context, atr)

        return None

    def should_close_position(self, df: pd.DataFrame, current_price: float,
                             position: Trade) -> bool:
        """Проверка выхода для соответствующей стратегии"""

        if position.side == PositionSide.LONG:
            return self.long_strategy.should_close_position(df, current_price, position)
        elif position.side == PositionSide.SHORT:
            return self.short_strategy.should_close_position(df, current_price, position)
        else:
            return self.neutral_strategy.should_close_position(df, current_price, position)


# """
# strategies.py - Реализация трёх торговых стратегий
# 1. Лонг-стратегия - торговля на откатах в восходящем тренде
# 2. Шорт-стратегия - торговля на откатах в нисходящем тренде
# 3. Нейтральная стратегия - диапазонная торговля
# """

# import pandas as pd
# import numpy as np
# from typing import Optional

# from trading_engine import TradingStrategy, Signal, OrderType, Trade, PositionSide
# from market_context import MarketContext
# from indicators import TechnicalIndicators
# from config import config

# class LongPullbackStrategy(TradingStrategy):
#     """
#     Лонг-стратегия: Торговля на откатах в восходящем тренде

#     Логика:
#     1. Работает только в BULLISH контексте
#     2. Ищет откаты к уровням Фибоначчи (38.2%, 50%, 61.8%)
#     3. Входит в лонг на откате
#     4. Stop-loss = цена входа - (ATR * 2)
#     5. Take-profit = цена входа + (ATR * 3.5)
#     """

#     def __init__(self):
#         self.last_swing_high = None
#         self.last_swing_low = None
#         self.looking_for_entry = False

#     def on_candle(self, df: pd.DataFrame, current_candle: pd.Series,
#                   context: MarketContext, atr: float) -> Optional[Signal]:
#         """Обработка новой свечи для лонг-стратегии"""

#         # Работаем только в бычьем контексте
#         if context != MarketContext.BULLISH:
#             self.looking_for_entry = False
#             return None

#         if atr == 0 or len(df) < 20:
#             return None

#         current_price = current_candle['close']

#         # Ищем последние swing high/low
#         swing_highs, swing_lows = TechnicalIndicators.find_swing_points(df, lookback=5)

#         # Обновляем последний максимум
#         if swing_highs.iloc[-6:-1].any():  # Проверяем предыдущие 5 свечей
#             idx = swing_highs.iloc[-6:-1].idxmax() if swing_highs.iloc[-6:-1].any() else None
#             if idx is not None:
#                 self.last_swing_high = df.loc[idx, 'high']

#         # Обновляем последний минимум  
#         if swing_lows.iloc[-6:-1].any():
#             idx = swing_lows.iloc[-6:-1].idxmin() if swing_lows.iloc[-6:-1].any() else None
#             if idx is not None:
#                 self.last_swing_low = df.loc[idx, 'low']

#         # Проверяем, есть ли откат
#         if self.last_swing_high is not None and self.last_swing_low is not None:
#             if self.last_swing_high > self.last_swing_low:  # Восходящее движение

#                 # Рассчитываем уровни Фибоначчи
#                 fib_levels = TechnicalIndicators.calculate_fibonacci_levels(
#                     self.last_swing_low, self.last_swing_high, is_uptrend=True
#                 )

#                 # Проверяем, достигли ли мы уровня 50% или 61.8%
#                 fib_50 = fib_levels['level_50']
#                 fib_618 = fib_levels['level_61.8']

#                 # Откат к уровню 61.8% (более глубокий откат)
#                 if current_price <= fib_618 and current_price >= self.last_swing_low:
#                     # Дополнительная проверка: цена должна снова пойти вверх (подтверждение)
#                     if len(df) >= 2 and current_candle['close'] > df.iloc[-2]['close']:

#                         # Генерируем сигнал на вход
#                         stop_loss = current_price - (atr * config.STOP_LOSS_ATR_MULTIPLIER)
#                         take_profit = current_price + (atr * config.TAKE_PROFIT_ATR_MULTIPLIER)

#                         return Signal(
#                             action=OrderType.BUY,
#                             price=current_price,
#                             stop_loss=stop_loss,
#                             take_profit=take_profit,
#                             reason=f"long_pullback_fib61.8"
#                         )

#                 # Откат к уровню 50%
#                 elif current_price <= fib_50 and current_price >= fib_618:
#                     if len(df) >= 2 and current_candle['close'] > df.iloc[-2]['close']:

#                         stop_loss = current_price - (atr * config.STOP_LOSS_ATR_MULTIPLIER)
#                         take_profit = current_price + (atr * config.TAKE_PROFIT_ATR_MULTIPLIER)

#                         return Signal(
#                             action=OrderType.BUY,
#                             price=current_price,
#                             stop_loss=stop_loss,
#                             take_profit=take_profit,
#                             reason=f"long_pullback_fib50"
#                         )

#         return None

#     def should_close_position(self, df: pd.DataFrame, current_price: float,
#                              position: Trade) -> bool:
#         """Проверка, нужно ли закрыть лонг позицию"""
#         # Закрываем только по stop/take, не закрываем досрочно
#         return False


# class ShortPullbackStrategy(TradingStrategy):
#     """
#     Шорт-стратегия: Торговля на откатах в нисходящем тренде

#     Логика:
#     1. Работает только в BEARISH контексте
#     2. Ищет откаты к уровням Фибоначчи (38.2%, 50%, 61.8%)
#     3. Входит в шорт на откате
#     4. Stop-loss = цена входа + (ATR * 2)
#     5. Take-profit = цена входа - (ATR * 3.5)
#     """

#     def __init__(self):
#         self.last_swing_high = None
#         self.last_swing_low = None

#     def on_candle(self, df: pd.DataFrame, current_candle: pd.Series,
#                   context: MarketContext, atr: float) -> Optional[Signal]:
#         """Обработка новой свечи для шорт-стратегии"""

#         # Работаем только в медвежьем контексте
#         if context != MarketContext.BEARISH:
#             return None

#         if atr == 0 or len(df) < 20:
#             return None

#         current_price = current_candle['close']

#         # Ищем последние swing high/low
#         swing_highs, swing_lows = TechnicalIndicators.find_swing_points(df, lookback=5)

#         # Обновляем последний максимум
#         if swing_highs.iloc[-6:-1].any():
#             idx = swing_highs.iloc[-6:-1].idxmax() if swing_highs.iloc[-6:-1].any() else None
#             if idx is not None:
#                 self.last_swing_high = df.loc[idx, 'high']

#         # Обновляем последний минимум
#         if swing_lows.iloc[-6:-1].any():
#             idx = swing_lows.iloc[-6:-1].idxmin() if swing_lows.iloc[-6:-1].any() else None
#             if idx is not None:
#                 self.last_swing_low = df.loc[idx, 'low']

#         # Проверяем, есть ли откат вверх
#         if self.last_swing_high is not None and self.last_swing_low is not None:
#             if self.last_swing_low < self.last_swing_high:  # Нисходящее движение

#                 # Рассчитываем уровни Фибоначчи для нисходящего движения
#                 fib_levels = TechnicalIndicators.calculate_fibonacci_levels(
#                     self.last_swing_high, self.last_swing_low, is_uptrend=False
#                 )

#                 # Для шорта - откат вверх
#                 # В нисходящем тренде откат это движение вверх
#                 # Поэтому используем обратную логику
#                 movement_range = self.last_swing_high - self.last_swing_low
#                 fib_50_level = self.last_swing_low + movement_range * 0.5
#                 fib_618_level = self.last_swing_low + movement_range * 0.618

#                 # Откат к уровню 61.8%
#                 if current_price >= fib_618_level and current_price <= self.last_swing_high:
#                     # Подтверждение: цена снова идёт вниз
#                     if len(df) >= 2 and current_candle['close'] < df.iloc[-2]['close']:

#                         stop_loss = current_price + (atr * config.STOP_LOSS_ATR_MULTIPLIER)
#                         take_profit = current_price - (atr * config.TAKE_PROFIT_ATR_MULTIPLIER)

#                         return Signal(
#                             action=OrderType.SELL,
#                             price=current_price,
#                             stop_loss=stop_loss,
#                             take_profit=take_profit,
#                             reason=f"short_pullback_fib61.8"
#                         )

#                 # Откат к уровню 50%
#                 elif current_price >= fib_50_level and current_price <= fib_618_level:
#                     if len(df) >= 2 and current_candle['close'] < df.iloc[-2]['close']:

#                         stop_loss = current_price + (atr * config.STOP_LOSS_ATR_MULTIPLIER)
#                         take_profit = current_price - (atr * config.TAKE_PROFIT_ATR_MULTIPLIER)

#                         return Signal(
#                             action=OrderType.SELL,
#                             price=current_price,
#                             stop_loss=stop_loss,
#                             take_profit=take_profit,
#                             reason=f"short_pullback_fib50"
#                         )

#         return None

#     def should_close_position(self, df: pd.DataFrame, current_price: float,
#                              position: Trade) -> bool:
#         """Проверка, нужно ли закрыть шорт позицию"""
#         return False


# class NeutralRangeStrategy(TradingStrategy):
#     """
#     Нейтральная стратегия: Диапазонная торговля

#     Логика:
#     1. Работает только в NEUTRAL контексте
#     2. Определяет диапазон дня (min/max за последние N дней)
#     3. Покупает около минимума диапазона
#     4. Продаёт около максимума диапазона
#     5. Stop-loss за границами диапазона
#     """

#     def __init__(self):
#         self.range_high = None
#         self.range_low = None
#         self.last_update_day = None

#     def _update_range(self, df: pd.DataFrame):
#         """Обновляет диапазон торговли на основе последних дней"""
#         if len(df) < config.RANGE_DAYS_LOOKBACK * 390:  # Примерно минут в торговом дне
#             return

#         # Берём последние N дней
#         lookback_candles = config.RANGE_DAYS_LOOKBACK * 390
#         recent_data = df.iloc[-lookback_candles:]

#         self.range_high = recent_data['high'].max()
#         self.range_low = recent_data['low'].min()

#     def on_candle(self, df: pd.DataFrame, current_candle: pd.Series,
#                   context: MarketContext, atr: float) -> Optional[Signal]:
#         """Обработка новой свечи для нейтральной стратегии"""

#         # Работаем только в нейтральном контексте
#         if context != MarketContext.NEUTRAL:
#             return None

#         if atr == 0:
#             return None

#         # Обновляем диапазон
#         self._update_range(df)

#         if self.range_high is None or self.range_low is None:
#             return None

#         current_price = current_candle['close']

#         # Отступ от границ диапазона
#         offset = atr * config.RANGE_ENTRY_OFFSET_ATR
#         buy_level = self.range_low + offset
#         sell_level = self.range_high - offset

#         # Сигнал на покупку около минимума
#         if current_price <= buy_level:
#             # Подтверждение: цена отскакивает вверх
#             if len(df) >= 2 and current_candle['close'] > df.iloc[-2]['low']:

#                 stop_loss = self.range_low - (atr * 0.5)  # Стоп чуть ниже минимума
#                 take_profit = sell_level  # Цель = верхняя граница диапазона

#                 return Signal(
#                     action=OrderType.BUY,
#                     price=current_price,
#                     stop_loss=stop_loss,
#                     take_profit=take_profit,
#                     reason="neutral_range_buy_low"
#                 )

#         # Сигнал на продажу около максимума
#         elif current_price >= sell_level:
#             # Подтверждение: цена отскакивает вниз
#             if len(df) >= 2 and current_candle['close'] < df.iloc[-2]['high']:

#                 stop_loss = self.range_high + (atr * 0.5)  # Стоп чуть выше максимума
#                 take_profit = buy_level  # Цель = нижняя граница диапазона

#                 return Signal(
#                     action=OrderType.SELL,
#                     price=current_price,
#                     stop_loss=stop_loss,
#                     take_profit=take_profit,
#                     reason="neutral_range_sell_high"
#                 )

#         return None

#     def should_close_position(self, df: pd.DataFrame, current_price: float,
#                              position: Trade) -> bool:
#         """Проверка, нужно ли закрыть позицию в нейтральной стратегии"""
#         # Закрываем если цена вышла за границы диапазона значительно
#         if self.range_high is None or self.range_low is None:
#             return False

#         # Для лонга: закрываем если пробили минимум диапазона вниз
#         if position.side == PositionSide.LONG:
#             if current_price < self.range_low * 0.995:  # -0.5% от минимума
#                 return True

#         # Для шорта: закрываем если пробили максимум диапазона вверх
#         if position.side == PositionSide.SHORT:
#             if current_price > self.range_high * 1.005:  # +0.5% от максимума
#                 return True

#         return False


# class CombinedStrategy(TradingStrategy):
#     """
#     Комбинированная стратегия - использует все три стратегии
#     в зависимости от контекста рынка
#     """

#     def __init__(self):
#         self.long_strategy = LongPullbackStrategy()
#         self.short_strategy = ShortPullbackStrategy()
#         self.neutral_strategy = NeutralRangeStrategy()

#     def on_candle(self, df: pd.DataFrame, current_candle: pd.Series,
#                   context: MarketContext, atr: float) -> Optional[Signal]:
#         """Выбирает стратегию в зависимости от контекста"""

#         if context == MarketContext.BULLISH:
#             return self.long_strategy.on_candle(df, current_candle, context, atr)

#         elif context == MarketContext.BEARISH:
#             return self.short_strategy.on_candle(df, current_candle, context, atr)

#         elif context == MarketContext.NEUTRAL:
#             return self.neutral_strategy.on_candle(df, current_candle, context, atr)

#         return None

#     def should_close_position(self, df: pd.DataFrame, current_price: float,
#                              position: Trade) -> bool:
#         """Проверка выхода для соответствующей стратегии"""

#         if position.side == PositionSide.LONG:
#             return self.long_strategy.should_close_position(df, current_price, position)
#         elif position.side == PositionSide.SHORT:
#             return self.short_strategy.should_close_position(df, current_price, position)
#         else:
#             return self.neutral_strategy.should_close_position(df, current_price, position)
