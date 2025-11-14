"""
trading_engine.py - Торговый движок с интерфейсом для стратегий
Реализует основную логику обработки свечей и принятия торговых решений
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
from enum import Enum

from config import config
from market_context import MarketContext, MarketContextAnalyzer
from indicators import TechnicalIndicators

class OrderType(Enum):
    """Тип ордера"""
    BUY = "buy"
    SELL = "sell"

class PositionSide(Enum):
    """Сторона позиции"""
    LONG = "long"
    SHORT = "short"
    NONE = "none"

@dataclass
class Trade:
    """Информация о сделке"""
    entry_time: datetime
    entry_price: float
    side: PositionSide
    stop_loss: float
    take_profit: float
    size: int = 1
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None  # 'stop_loss', 'take_profit', 'force_close', 'strategy'

@dataclass
class Signal:
    """Торговый сигнал"""
    action: OrderType
    price: float
    stop_loss: float
    take_profit: float
    reason: str

class TradingStrategy(ABC):
    """Абстрактный базовый класс для торговых стратегий"""

    @abstractmethod
    def on_candle(self, df: pd.DataFrame, current_candle: pd.Series, 
                  context: MarketContext, atr: float) -> Optional[Signal]:
        """
        Обработка новой свечи - основная логика стратегии

        Args:
            df: Исторические данные
            current_candle: Текущая свеча
            context: Текущий контекст рынка
            atr: Текущее значение ATR

        Returns:
            Signal или None
        """
        pass

    @abstractmethod
    def should_close_position(self, df: pd.DataFrame, current_price: float,
                             position: Trade) -> bool:
        """
        Проверка, нужно ли закрыть текущую позицию

        Args:
            df: Исторические данные
            current_price: Текущая цена
            position: Открытая позиция

        Returns:
            True если позицию нужно закрыть
        """
        pass

class TradingEngine:
    """
    Главный торговый движок
    Координирует работу стратегий, контекста и управление позициями
    """

    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
        self.context_analyzer = MarketContextAnalyzer()
        self.current_position: Optional[Trade] = None
        self.trades_history: List[Trade] = []
        self.current_context = MarketContext.UNKNOWN

        # Состояние кулдауна после stop-loss
        self.in_cooldown = False
        self.cooldown_until: Optional[datetime] = None
        self.cooldown_price_target: Optional[float] = None

        # Данные для расчётов
        self.historical_data = pd.DataFrame()
        self.current_atr = 0.0

    def on_candle_received(self, df: pd.DataFrame, current_time: datetime):
        """
        ГЛАВНЫЙ МЕТОД: Обработка поступления новой свечи
        Вся логика принятия решений происходит здесь

        Args:
            df: DataFrame с историческими данными
            current_time: Время текущей свечи
        """
        self.historical_data = df
        current_candle = df.iloc[-1]
        current_price = current_candle['close']

        # 1. ПРОВЕРКА ВРЕМЕНИ - ОБЯЗАТЕЛЬНОЕ ЗАКРЫТИЕ ПОЗИЦИЙ НА НОЧЬ!
        if not self._is_trading_time(current_time):
            if self.current_position is not None:
                self._close_position(current_price, current_time, "force_close_night")
                print(f"⚠️ ПРИНУДИТЕЛЬНОЕ ЗАКРЫТИЕ позиции на ночь! Время: {current_time}")
            return

        # 2. Рассчитываем индикаторы
        if len(df) >= config.ATR_PERIOD:
            atr_series = TechnicalIndicators.calculate_atr(df, config.ATR_PERIOD)
            self.current_atr = atr_series.iloc[-1] if not pd.isna(atr_series.iloc[-1]) else 0.0

        # 3. Обновляем контекст рынка
        prev_context = self.current_context
        self.current_context = self.context_analyzer.update_context(df)

        # Если контекст изменился, вызываем обработчик
        if prev_context != self.current_context and prev_context != MarketContext.UNKNOWN:
            self.on_context_changed(self.current_context)

        # 4. Проверяем кулдаун
        if self.in_cooldown:
            self._check_cooldown(current_price, current_time)
            if self.in_cooldown:  # Если всё ещё в кулдауне - не торгуем
                return

        # 5. Управление открытой позицией
        if self.current_position is not None:
            # Проверяем stop-loss и take-profit
            self._check_exit_conditions(current_price, current_time)

            # Проверяем условия стратегии для выхода
            if self.current_position is not None:  # Может быть закрыта выше
                if self.strategy.should_close_position(df, current_price, self.current_position):
                    self._close_position(current_price, current_time, "strategy_signal")

        # 6. Ищем новые входы (если нет открытой позиции)
        if self.current_position is None and not self.in_cooldown:
            signal = self.strategy.on_candle(df, current_candle, self.current_context, self.current_atr)

            if signal is not None:
                self._open_position(signal, current_time)

    def on_context_changed(self, new_context: MarketContext):
        """
        Обработка изменения контекста рынка
        Может закрыть позицию, если она не соответствует новому контексту

        Args:
            new_context: Новый контекст рынка
        """
        print(f"📊 Контекст рынка изменился на: {new_context.value}")
        # Для Breakout стратегий НЕ закрываем по контексту
        strategy_name = self.strategy.__class__.__name__
        if 'Breakout' in strategy_name:
            return  # Breakout не зависит от контекста
        
        # Для стратегий откатов ЗАКРЫВАЕМ при несоответствии контекста
        if self.current_position is not None:
            current_price = self.historical_data['close'].iloc[-1]
            current_time = self.historical_data.index[-1]
            
            if (self.current_position.side == PositionSide.LONG and 
                new_context == MarketContext.BEARISH):
                self._close_position(current_price, current_time, "context_change")
            
            elif (self.current_position.side == PositionSide.SHORT and 
                new_context == MarketContext.BULLISH):
                self._close_position(current_price, current_time, "context_change")

        # # Если есть открытая позиция, проверяем соответствие контексту
        # if self.current_position is not None:
        #     current_price = self.historical_data['close'].iloc[-1]
        #     current_time = self.historical_data.index[-1]

        #     # Лонг позиция в шорт контексте - закрываем
        #     if (self.current_position.side == PositionSide.LONG and 
        #         new_context == MarketContext.BEARISH):
        #         self._close_position(current_price, current_time, "context_change")

        #     # Шорт позиция в лонг контексте - закрываем
        #     elif (self.current_position.side == PositionSide.SHORT and 
        #           new_context == MarketContext.BULLISH):
        #         self._close_position(current_price, current_time, "context_change")

    def _is_trading_time(self, current_time: datetime) -> bool:
        """Проверяет, находимся ли мы во время торговли"""
        current_time_only = current_time.time()

        # Принудительное закрытие перед ночью
        if current_time_only >= config.FORCE_CLOSE_TIME:
            return False

        # Обычное торговое время
        return config.TRADING_START_TIME <= current_time_only <= config.TRADING_END_TIME

    def _open_position(self, signal: Signal, current_time: datetime):
        """Открывает новую позицию"""
        side = PositionSide.LONG if signal.action == OrderType.BUY else PositionSide.SHORT

        self.current_position = Trade(
            entry_time=current_time,
            entry_price=signal.price,
            side=side,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            size=config.POSITION_SIZE
        )

        print(f"✅ Открыта {side.value} позиция: цена={signal.price:.2f}, "
              f"SL={signal.stop_loss:.2f}, TP={signal.take_profit:.2f}, причина={signal.reason}")

    def _close_position(self, exit_price: float, exit_time: datetime, reason: str):
        """Закрывает текущую позицию"""
        if self.current_position is None:
            return

        # Рассчитываем PnL
        if self.current_position.side == PositionSide.LONG:
            pnl = (exit_price - self.current_position.entry_price) * self.current_position.size
        else:  # SHORT
            pnl = (self.current_position.entry_price - exit_price) * self.current_position.size

        # Учитываем комиссию
        commission = (self.current_position.entry_price + exit_price) * config.COMMISSION
        pnl -= commission

        self.current_position.exit_time = exit_time
        self.current_position.exit_price = exit_price
        self.current_position.pnl = pnl
        self.current_position.exit_reason = reason

        # Добавляем в историю
        self.trades_history.append(self.current_position)

        print(f"❌ Закрыта {self.current_position.side.value} позиция: "
              f"цена={exit_price:.2f}, PnL={pnl:.2f}, причина={reason}")

        # Если закрытие по stop-loss - включаем кулдаун
        if reason == "stop_loss":
            self._activate_cooldown(exit_price, exit_time)

        self.current_position = None

    def _check_exit_conditions(self, current_price: float, current_time: datetime):
        """Проверяет условия выхода (stop-loss и take-profit)"""
        if self.current_position is None:
            return

        # Stop-loss
        if self.current_position.side == PositionSide.LONG:
            if current_price <= self.current_position.stop_loss:
                self._close_position(current_price, current_time, "stop_loss")
                return
            if current_price >= self.current_position.take_profit:
                self._close_position(current_price, current_time, "take_profit")
                return
        else:  # SHORT
            if current_price >= self.current_position.stop_loss:
                self._close_position(current_price, current_time, "stop_loss")
                return
            if current_price <= self.current_position.take_profit:
                self._close_position(current_price, current_time, "take_profit")
                return

    def _activate_cooldown(self, stop_price: float, current_time: datetime):
        """Активирует кулдаун после срабатывания stop-loss"""
        self.in_cooldown = True
        self.cooldown_until = current_time + timedelta(minutes=config.COOLDOWN_MINUTES)

        # Ценовой таргет для выхода из кулдауна
        cooldown_distance = self.current_atr * config.COOLDOWN_ATR_MULTIPLIER

        # Зависит от направления последней позиции
        last_trade = self.trades_history[-1]
        if last_trade.side == PositionSide.LONG:
            # Для лонга ждём дальнейшего падения
            self.cooldown_price_target = stop_price - cooldown_distance
        else:
            # Для шорта ждём дальнейшего роста
            self.cooldown_price_target = stop_price + cooldown_distance

        print(f"⏸️ КУЛДАУН активирован до {self.cooldown_until}, "
              f"ценовой таргет: {self.cooldown_price_target:.2f}")

    def _check_cooldown(self, current_price: float, current_time: datetime):
        """Проверяет, можно ли выйти из кулдауна"""
        # Временной кулдаун истёк
        if current_time >= self.cooldown_until:
            self.in_cooldown = False
            print(f"✅ Кулдаун завершён (время)")
            return

        # Ценовой таргет достигнут
        last_trade = self.trades_history[-1]
        if last_trade.side == PositionSide.LONG:
            if current_price <= self.cooldown_price_target:
                self.in_cooldown = False
                print(f"✅ Кулдаун завершён (цена достигла таргета)")
        else:
            if current_price >= self.cooldown_price_target:
                self.in_cooldown = False
                print(f"✅ Кулдаун завершён (цена достигла таргета)")

    def get_statistics(self) -> dict:
        """Возвращает статистику по сделкам"""
        if not self.trades_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_win': 0,
                'max_loss': 0
            }

        wins = [t.pnl for t in self.trades_history if t.pnl > 0]
        losses = [t.pnl for t in self.trades_history if t.pnl < 0]

        return {
            'total_trades': len(self.trades_history),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(self.trades_history) * 100 if self.trades_history else 0,
            'total_pnl': sum(t.pnl for t in self.trades_history),
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'max_win': max(wins) if wins else 0,
            'max_loss': min(losses) if losses else 0
        }
