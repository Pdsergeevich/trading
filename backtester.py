"""
backtester.py - Модуль для бэктестинга стратегий на исторических данных
Включает визуализацию сделок и анализ результатов
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from trading_engine import TradingEngine, TradingStrategy, Trade, PositionSide
from config import config

class Backtester:
    """
    Класс для бэктестинга торговых стратегий
    Прогоняет стратегию по историческим данным и собирает статистику
    """

    def __init__(self, strategy: TradingStrategy, data: pd.DataFrame):
        """
        Инициализация бэктестера

        Args:
            strategy: Торговая стратегия для тестирования
            data: Исторические данные (DataFrame с OHLCV)
        """
        self.strategy = strategy
        self.data = data.copy()
        self.engine = TradingEngine(strategy)
        self.interrupted = False

        # Убедимся что индекс - datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'timestamp' in self.data.columns:
                self.data.index = pd.to_datetime(self.data['timestamp'])
            elif 'time' in self.data.columns:
                self.data.index = pd.to_datetime(self.data['time'])

        self.results = None

    def run(self, start_date: datetime = None, end_date: datetime = None) -> dict:
        """
        Запускает бэктест на исторических данных

        Args:
            start_date: Дата начала бэктеста (опционально)
            end_date: Дата окончания бэктеста (опционально)

        Returns:
            Словарь с результатами бэктеста
        """
        print("🚀 Запуск бэктестинга...")
        print(f"Период: {self.data.index[0]} - {self.data.index[-1]}")
        print(f"Всего свечей: {len(self.data)}")

        # Фильтруем данные по датам если указаны
        test_data = self.data.copy()
        if start_date:
            test_data = test_data[test_data.index >= start_date]
        if end_date:
            test_data = test_data[test_data.index <= end_date]

        # Прогоняем каждую свечу через движок
        for i in range(len(test_data)):

            if self.interrupted:
                print("\n⚠️ Бэктестинг прерван")
                break
            # Берём данные до текущей свечи (включительно)
            current_data = test_data.iloc[:i+1]
            current_time = test_data.index[i]

            # Передаём в движок
            self.engine.on_candle_received(current_data, current_time)

        # Принудительно закрываем открытую позицию в конце
        if self.engine.current_position is not None:
            last_price = test_data['close'].iloc[-1]
            last_time = test_data.index[-1]
            self.engine._close_position(last_price, last_time, "backtest_end")

        # Собираем статистику
        stats = self.engine.get_statistics()

        # Рассчитываем дополнительные метрики
        self.results = self._calculate_metrics(stats)

        print("\n✅ Бэктестинг завершён!")
        self._print_results()

        return self.results

    def _calculate_metrics(self, stats: dict) -> dict:
        """Рассчитывает расширенные метрики"""

        trades = self.engine.trades_history

        if not trades:
            return {
                **stats,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'avg_trade_duration': 0,
                'final_capital': config.INITIAL_CAPITAL
            }

        # Equity curve (кривая капитала)
        equity = [config.INITIAL_CAPITAL]
        for trade in trades:
            equity.append(equity[-1] + trade.pnl)

        # Максимальная просадка
        peak = equity[0]
        max_dd = 0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Profit Factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe Ratio (упрощённый)
        returns = [t.pnl for t in trades]
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Средняя длительность сделки
        durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]
        avg_duration = np.mean(durations) if durations else 0

        return {
            **stats,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'avg_trade_duration_minutes': avg_duration,
            'final_capital': equity[-1],
            'equity_curve': equity
        }

    def _print_results(self):
        """Выводит результаты бэктестинга"""
        r = self.results

        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ БЭКТЕСТИНГА")
        print("="*60)
        print(f"Всего сделок:        {r['total_trades']}")
        print(f"Прибыльных сделок:   {r['winning_trades']} ({r['win_rate']:.1f}%)")
        print(f"Убыточных сделок:    {r['losing_trades']}")
        print(f"\nОбщий PnL:           {r['total_pnl']:.2f} руб")
        print(f"Средняя прибыль:     {r['avg_win']:.2f} руб")
        print(f"Средний убыток:      {r['avg_loss']:.2f} руб")
        print(f"Макс. прибыль:       {r['max_win']:.2f} руб")
        print(f"Макс. убыток:        {r['max_loss']:.2f} руб")
        print(f"\nProfit Factor:       {r['profit_factor']:.2f}")
        print(f"Sharpe Ratio:        {r['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:        {r['max_drawdown']:.2f}%")
        print(f"\nНачальный капитал:   {config.INITIAL_CAPITAL:.2f} руб")
        print(f"Конечный капитал:    {r['final_capital']:.2f} руб")
        print(f"Доходность:          {((r['final_capital']/config.INITIAL_CAPITAL - 1) * 100):.2f}%")
        print(f"\nСредняя длительность сделки: {r['avg_trade_duration_minutes']:.0f} минут")
        print("="*60)

    def plot_results(self, save_path: str = 'backtest_results.png'):
        """
        Визуализирует результаты бэктестинга
        Показывает график цены с точками входа/выхода и индикаторами

        Args:
            save_path: Путь для сохранения графика
        """
        if self.results is None:
            print("❌ Сначала запустите бэктест методом run()")
            return

        trades = self.engine.trades_history

        # Создаём фигуру с несколькими подграфиками
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                            gridspec_kw={'height_ratios': [3, 1, 1]})

        # График 1: Цена + сделки
        ax1.plot(self.data.index, self.data['close'], label='Цена закрытия', 
                linewidth=1, color='black', alpha=0.7)

        # Отмечаем входы и выходы
        for trade in trades:
            # Вход
            color = 'green' if trade.side == PositionSide.LONG else 'red'
            marker = '^' if trade.side == PositionSide.LONG else 'v'
            ax1.scatter(trade.entry_time, trade.entry_price, 
                       color=color, marker=marker, s=100, zorder=5,
                       label='Long вход' if trade == trades[0] and trade.side == PositionSide.LONG else 
                             'Short вход' if trade == trades[0] and trade.side == PositionSide.SHORT else '')

            # Выход
            exit_color = 'darkgreen' if trade.pnl > 0 else 'darkred'
            ax1.scatter(trade.exit_time, trade.exit_price,
                       color=exit_color, marker='x', s=100, zorder=5,
                       label='Выход (прибыль)' if trade == trades[0] and trade.pnl > 0 else
                             'Выход (убыток)' if trade == trades[0] and trade.pnl < 0 else '')

            # Линия сделки
            ax1.plot([trade.entry_time, trade.exit_time],
                    [trade.entry_price, trade.exit_price],
                    color=exit_color, linestyle='--', alpha=0.3, linewidth=1)

        # Добавляем EMA если есть
        if self.engine.context_analyzer.ema_fast is not None:
            ax1.plot(self.data.index, self.engine.context_analyzer.ema_fast,
                    label=f'EMA {config.EMA_FAST}', alpha=0.5, linewidth=1)
            ax1.plot(self.data.index, self.engine.context_analyzer.ema_slow,
                    label=f'EMA {config.EMA_SLOW}', alpha=0.5, linewidth=1)

        ax1.set_ylabel('Цена', fontsize=12)
        ax1.set_title('Результаты бэктестинга: Цена и сделки', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # График 2: Equity curve (кривая капитала)
        equity = self.results['equity_curve']
        equity_times = [self.data.index[0]] + [t.exit_time for t in trades]
        ax2.plot(equity_times, equity, color='blue', linewidth=2)
        ax2.fill_between(equity_times, config.INITIAL_CAPITAL, equity, 
                        where=np.array(equity) >= config.INITIAL_CAPITAL,
                        color='green', alpha=0.3, label='Прибыль')
        ax2.fill_between(equity_times, config.INITIAL_CAPITAL, equity,
                        where=np.array(equity) < config.INITIAL_CAPITAL,
                        color='red', alpha=0.3, label='Убыток')
        ax2.axhline(y=config.INITIAL_CAPITAL, color='black', linestyle='--', 
                   linewidth=1, alpha=0.5, label='Начальный капитал')
        ax2.set_ylabel('Капитал (руб)', fontsize=12)
        ax2.set_title('Кривая капитала', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # График 3: ADX (сила тренда)
        if self.engine.context_analyzer.adx is not None:
            ax3.plot(self.data.index, self.engine.context_analyzer.adx,
                    label='ADX', color='purple', linewidth=1.5)
            ax3.axhline(y=config.ADX_TREND_THRESHOLD, color='green', 
                       linestyle='--', alpha=0.5, label='Порог тренда (25)')
            ax3.axhline(y=config.ADX_NEUTRAL_THRESHOLD, color='orange',
                       linestyle='--', alpha=0.5, label='Порог нейтрального (20)')
            ax3.fill_between(self.data.index, 0, self.engine.context_analyzer.adx,
                           where=self.engine.context_analyzer.adx >= config.ADX_TREND_THRESHOLD,
                           color='green', alpha=0.2)
            ax3.fill_between(self.data.index, 0, self.engine.context_analyzer.adx,
                           where=self.engine.context_analyzer.adx < config.ADX_NEUTRAL_THRESHOLD,
                           color='orange', alpha=0.2)
            ax3.set_ylabel('ADX', fontsize=12)
            ax3.set_xlabel('Время', fontsize=12)
            ax3.set_title('Индикатор силы тренда (ADX)', fontsize=12)
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 График сохранён: {save_path}")
        plt.close()

    def get_trade_log(self) -> pd.DataFrame:
        """
        Возвращает детальный лог всех сделок

        Returns:
            DataFrame с информацией о сделках
        """
        trades = self.engine.trades_history

        if not trades:
            return pd.DataFrame()

        trade_data = []
        for i, trade in enumerate(trades, 1):
            trade_data.append({
                'trade_num': i,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'side': trade.side.value,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'pnl': trade.pnl,
                'pnl_percent': (trade.pnl / trade.entry_price) * 100,
                'duration_minutes': (trade.exit_time - trade.entry_time).total_seconds() / 60,
                'exit_reason': trade.exit_reason
            })

        return pd.DataFrame(trade_data)
