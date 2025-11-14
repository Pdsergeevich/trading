"""
main.py - Главный файл для запуска торгового бота
Работает с реальными историческими данными из CSV файлов
Поддерживает различные торговые стратегии и визуализацию результатов
"""

from datetime import datetime
import sys
import os
import glob
import pandas as pd

from config import config
from data_loader import DataLoader
from strategies import LongPullbackStrategy, ShortPullbackStrategy, NeutralRangeStrategy, CombinedStrategy
from backtester import Backtester
from trading_engine import TradingEngine
from strategies_futures import BreakoutStrategy
from strategies_ml import MLEnhancedBreakoutStrategy
from ml_predictor import MLPredictor

def run_backtest_with_real_data():
    """
    Главная функция для запуска бэктестинга на реальных исторических данных
    
    Этапы работы:
    1. Поиск CSV файлов с историческими данными
    2. Загрузка выбранного файла
    3. Выбор торговой стратегии
    4. Запуск бэктестинга
    5. Вывод результатов и сохранение графиков
    """
    
    print("="*70)
    print("🎯 БЭКТЕСТИНГ НА РЕАЛЬНЫХ ДАННЫХ")
    print("="*70)
    
    # ========================================================================
    # ШАГ 1: ПОИСК И ВЫБОР ФАЙЛА С ДАННЫМИ
    # ========================================================================
    print("\n1️⃣ ПОИСК ФАЙЛОВ С ДАННЫМИ")
    print("-" * 70)
    
    # Ищем все CSV файлы, начинающиеся с "candles_"
    csv_files = glob.glob("candles_*.csv")
    csv_files.sort()  # Сортируем по имени
    
    if csv_files:
        print(f"\n✅ Найдено {len(csv_files)} файл(ов) с данными:\n")
        
        # Показываем список всех файлов с размером
        for i, file in enumerate(csv_files, 1):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"   {i}. {file} ({size_mb:.2f} MB)")
        
        print("\n" + "-" * 70)
        
        # Выбор файла пользователем
        choice = input(f"\nВыберите файл (1-{len(csv_files)}) или нажмите Enter для первого: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
            selected_file = csv_files[int(choice) - 1]
        else:
            selected_file = csv_files[0]  # По умолчанию первый файл
        
        print(f"\n✅ Выбран файл: {selected_file}")
        
    else:
        # Если файлов нет - предлагаем использовать тестовые данные
        print("\n⚠️ Файлы с данными не найдены!")
        print("\n💡 Сначала скачайте данные:")
        print("   python download_history.py")
        print("\nИли создайте CSV файл со свечами (колонки: timestamp, open, high, low, close, volume)")
        
        use_test = input("\nИспользовать тестовые синтетические данные? (y/n): ").strip().lower()
        
        if use_test == 'y':
            print("\n⚠️ Используются синтетические данные для демонстрации")
            data = DataLoader.generate_sample_data(days=30, interval_minutes=1, start_price=100000.0)
            data = DataLoader.filter_trading_hours(data, start_time="10:00", end_time="18:45")
            selected_file = "synthetic_data"
        else:
            print("\n❌ Бэктестинг отменён")
            return None
    
    # ========================================================================
    # ШАГ 2: ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
    # ========================================================================
    if selected_file != "synthetic_data":
        print("\n2️⃣ ЗАГРУЗКА ДАННЫХ")
        print("-" * 70)
        
        # Загружаем CSV файл
        data = DataLoader.load_from_csv(selected_file, date_column='timestamp')
        
        # Фильтруем только торговые часы (10:00-18:45 по МСК)
        print("\n🕐 Фильтрация торговых часов (10:00-18:45)...")
        data = DataLoader.filter_trading_hours(data, start_time="10:00", end_time="18:45")
    
    # ========================================================================
    # ШАГ 3: ВЫБОР ТОРГОВОЙ СТРАТЕГИИ
    # ========================================================================
    print("\n3️⃣ ВЫБОР СТРАТЕГИИ")
    print("-" * 70)
    print("\nДоступные стратегии:")
    print("   1. Long Pullback - Только лонг на откатах в восходящем тренде")
    print("   2. Short Pullback - Только шорт на откатах в нисходящем тренде")
    print("   3. Neutral Range - Диапазонная торговля в боковике")
    print("   4. Combined - Все три стратегии (автоматический выбор по контексту) ⭐")
    print("   5. Breakout - Пробой диапазона (ДЛЯ ФЬЮЧЕРСОВ) 🚀")
    print("   6. ML-Enhanced Breakout - С машинным обучением ⭐⭐⭐")
    
    strategy_choice = input("\nВыберите стратегию (1-6, по умолчанию 5): ").strip()
    
    # Создание объекта стратегии в зависимости от выбора
    if strategy_choice == "1":
        strategy = LongPullbackStrategy()
        strategy_name = "Long Pullback"
        
    elif strategy_choice == "2":
        strategy = ShortPullbackStrategy()
        strategy_name = "Short Pullback"
        
    elif strategy_choice == "3":
        strategy = NeutralRangeStrategy()
        strategy_name = "Neutral Range"
        
    elif strategy_choice == "4":
        strategy = CombinedStrategy()
        strategy_name = "Combined"
        
    elif strategy_choice == "6":
        # ===== ML-СТРАТЕГИЯ: УМНЫЙ ПОИСК МОДЕЛИ =====
        
        # Извлекаем тикер инструмента из выбранного файла
        # Например: candles_SBER_50days.csv → SBER
        #          candles_RIZ5_100days.csv → RIZ5
        instrument_name = selected_file.replace('candles_', '').split('_')[0]
        
        # Ищем ВСЕ модели для данного инструмента (любой период обучения)
        available_models = sorted(glob.glob(f'candles_{instrument_name}_*_ml_model.pkl'))
        
        if not available_models:
            # Модель не найдена - предлагаем обучить
            print(f"\n⚠️ ML-модель для {instrument_name} не найдена")
            print(f"\n💡 Создайте модель:")
            print(f"   1. Запустите: python main.py → опция 4 (Обучить ML)")
            print(f"   2. Выберите файл с данными {instrument_name}")
            print(f"   3. После обучения модель автоматически сохранится")
            print(f"\n   Или обучите вручную: python train_ml_model.py")
            return None
        
        # Если найдено несколько моделей - показываем меню выбора
        if len(available_models) > 1:
            print(f"\n📊 Найдено несколько моделей для {instrument_name}:\n")
            for i, model_path in enumerate(available_models, 1):
                # Показываем точность модели если доступна
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        accuracy = model_data.get('accuracy', 0) * 100
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    print(f"   {i}. {model_path} (точность: {accuracy:.1f}%, {size_mb:.2f} MB)")
                except:
                    print(f"   {i}. {model_path}")
            
            print("\n" + "-"*70)
            choice = input(f"Выберите модель (1-{len(available_models)}, Enter для первой): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                ml_model_path = available_models[int(choice) - 1]
            else:
                ml_model_path = available_models[0]  # По умолчанию первая
            
            print(f"\n✅ Используется модель: {ml_model_path}")
        
        else:
            # Найдена одна модель - используем её автоматически
            ml_model_path = available_models[0]
            print(f"\n✅ Найдена модель: {ml_model_path}")
        
        # Создаём стратегию с выбранной моделью
        strategy = MLEnhancedBreakoutStrategy(
            ml_model_path=ml_model_path,
            use_ml=True,
            min_confidence='MEDIUM'
        )
        strategy_name = "ML-Enhanced Breakout"
        
    else:
        # По умолчанию используем простую Breakout стратегию
        strategy = BreakoutStrategy()
        strategy_name = "Breakout"
    
    print(f"\n✅ Выбрана стратегия: {strategy_name}")
    
    # ========================================================================
    # ШАГ 4: ЗАПУСК БЭКТЕСТИНГА С ОБРАБОТКОЙ CTRL+C
    # ========================================================================
    print("\n4️⃣ ЗАПУСК БЭКТЕСТИНГА")
    print("-" * 70)
    print("⚠️ Для остановки нажмите Ctrl+C\n")
    
    # Создаём объект бэктестера
    backtester = Backtester(strategy=strategy, data=data)
    
    try:
        # ===== ОСНОВНОЙ ЗАПУСК БЭКТЕСТИНГА =====
        results = backtester.run()
        
        # ========================================================================
        # ШАГ 5: ВИЗУАЛИЗАЦИЯ И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
        # ========================================================================
        print("\n5️⃣ ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
        print("-" * 70)
        
        # Формируем имена файлов для сохранения
        instrument_name = selected_file.replace('candles_', '').replace('.csv', '')
        result_filename = f'backtest_{instrument_name}_{strategy_name.replace(" ", "_").lower()}.png'
        
        # Строим и сохраняем график
        backtester.plot_results(save_path=result_filename)
        
        # Сохраняем детальный лог сделок в CSV
        trade_log = backtester.get_trade_log()
        if not trade_log.empty:
            log_filename = f'trades_{instrument_name}_{strategy_name.replace(" ", "_").lower()}.csv'
            trade_log.to_csv(log_filename, index=False)
            print(f"📄 Детальный лог сделок: {log_filename}")
            
            # Показываем первые 10 сделок
            print("\n📊 ПЕРВЫЕ 10 СДЕЛОК:")
            print("-" * 70)
            print(trade_log.head(10).to_string(index=False))
        
        # ========================================================================
        # ШАГ 6: ИТОГОВАЯ СТАТИСТИКА И ОЦЕНКА
        # ========================================================================
        print("\n" + "="*70)
        print("📈 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("="*70)
        
        if results['total_trades'] > 0:
            print(f"\n✅ Стратегия: {strategy_name}")
            print(f"📁 Данные: {selected_file}")
            print(f"📊 График: {result_filename}")
            print(f"📄 Лог: {log_filename if not trade_log.empty else 'N/A'}")
            
            # Оценка результатов по ключевым метрикам
            print("\n" + "-"*70)
            print("ОЦЕНКА РЕЗУЛЬТАТОВ:")
            print("-"*70)
            
            # Win Rate
            if results['win_rate'] >= 55:
                print("✅ Win Rate отличный (≥55%)")
            elif results['win_rate'] >= 45:
                print("⚠️ Win Rate средний (45-55%)")
            else:
                print("❌ Win Rate низкий (<45%)")
            
            # Profit Factor
            if results['profit_factor'] >= 1.5:
                print("✅ Profit Factor отличный (≥1.5)")
            elif results['profit_factor'] >= 1.0:
                print("⚠️ Profit Factor средний (1.0-1.5)")
            else:
                print("❌ Profit Factor низкий (<1.0)")
            
            # Sharpe Ratio
            if results['sharpe_ratio'] >= 1.0:
                print("✅ Sharpe Ratio хороший (≥1.0)")
            elif results['sharpe_ratio'] >= 0.5:
                print("⚠️ Sharpe Ratio средний (0.5-1.0)")
            else:
                print("❌ Sharpe Ratio низкий (<0.5)")
            
            # Max Drawdown
            if results['max_drawdown'] <= 15:
                print("✅ Просадка приемлемая (≤15%)")
            elif results['max_drawdown'] <= 25:
                print("⚠️ Просадка высокая (15-25%)")
            else:
                print("❌ Просадка критическая (>25%)")
            
            # Рекомендации
            print("\n" + "-"*70)
            print("РЕКОМЕНДАЦИИ:")
            print("-"*70)
            
            if results['total_pnl'] > 0:
                print("✅ Стратегия прибыльна на исторических данных")
                print("\n💡 Следующие шаги:")
                print("   1. Протестируйте на других периодах времени")
                print("   2. Оптимизируйте параметры в config.py")
                print("   3. Проведите walk-forward анализ")
                print("   4. Запустите на forward testing (out-of-sample)")
            else:
                print("❌ Стратегия убыточна на этих данных")
                print("\n💡 Попробуйте:")
                print("   1. Изменить параметры в config.py:")
                print("      - STOP_LOSS_ATR_MULTIPLIER (сейчас 2.0)")
                print("      - TAKE_PROFIT_ATR_MULTIPLIER (сейчас 3.5)")
                print("      - COOLDOWN_MINUTES (сейчас 15)")
                print("   2. Протестировать другую стратегию")
                print("   3. Использовать другой инструмент")
        else:
            print("\n⚠️ Сделок не было!")
            print("\nВозможные причины:")
            print("   1. Недостаточно данных для расчёта индикаторов")
            print("   2. Контекст рынка не подходит для выбранной стратегии")
            print("   3. Параметры входа слишком строгие")
            print("\n💡 Попробуйте Combined стратегию или другой период данных")
        
        print("\n" + "="*70)
        return results, backtester
    
    # ========================================================================
    # ОБРАБОТКА ПРЕРЫВАНИЯ ПОЛЬЗОВАТЕЛЕМ (CTRL+C)
    # ========================================================================
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⚠️ ПОЛУЧЕН СИГНАЛ ПРЕРЫВАНИЯ (Ctrl+C)")
        print("="*70)
        print("🛑 Останавливаем бэктестинг...\n")
        
        # Закрываем открытую позицию, если она есть
        if backtester.engine.current_position:
            print("📌 Закрываем открытую позицию принудительно...")
            current_price = backtester.data.iloc[-1]['close']
            backtester.engine._close_position(
                current_price, 
                backtester.data.index[-1], 
                "force_exit"
            )
            print("   ✅ Позиция закрыта")
        
        # Выводим частичные результаты (сделки до момента остановки)
        print("\n📊 ЧАСТИЧНЫЕ РЕЗУЛЬТАТЫ (до момента остановки)")
        print("-"*70)
        
        if backtester.engine.trades_history:
            # Рассчитываем статистику по совершённым сделкам
            stats = backtester.engine.get_statistics()
            results = backtester._calculate_metrics(stats)
            backtester.results = results
            backtester._print_results()
            
            # Сохраняем частичные результаты в файлы
            print("\n💾 Сохранение частичных результатов...")
            instrument_name = selected_file.replace('candles_', '').replace('.csv', '')
            result_filename = f'backtest_{instrument_name}_{strategy_name.replace(" ", "_").lower()}_interrupted.png'
            
            try:
                backtester.plot_results(save_path=result_filename)
                print(f"   ✅ График сохранён: {result_filename}")
            except Exception as e:
                print(f"   ⚠️ Не удалось сохранить график: {e}")
            
            # Сохраняем CSV с логом сделок
            trade_log = backtester.get_trade_log()
            if not trade_log.empty:
                log_filename = f'trades_{instrument_name}_{strategy_name.replace(" ", "_").lower()}_interrupted.csv'
                trade_log.to_csv(log_filename, index=False)
                print(f"   ✅ Лог сделок сохранён: {log_filename}")
        else:
            print("   ⚠️ Сделок не было совершено")
        
        print("\n" + "="*70)
        print("✅ Программа корректно завершена")
        print("="*70)
        return None
    
    # ========================================================================
    # ОБРАБОТКА ДРУГИХ ОШИБОК
    # ========================================================================
    except Exception as e:
        print(f"\n❌ Ошибка во время выполнения: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Этот блок выполнится в любом случае (успех, прерывание или ошибка)
        print("\n🔚 Завершение работы...")


def compare_strategies_on_real_data():
    """
    Сравнивает несколько стратегий на одних и тех же данных
    Полезно для выбора лучшей стратегии для конкретного инструмента
    """
    print("="*70)
    print("🔬 СРАВНЕНИЕ ВСЕХ СТРАТЕГИЙ")
    print("="*70)
    
    # Поиск файлов с данными
    csv_files = glob.glob("candles_*.csv")
    
    if not csv_files:
        print("\n❌ Файлы с данными не найдены!")
        print("   Запустите: python download_history.py")
        return
    
    # Используем первый найденный файл
    selected_file = csv_files[0]
    print(f"\n📂 Используется: {selected_file}")
    
    # Загружаем данные
    data = DataLoader.load_from_csv(selected_file, date_column='timestamp')
    data = DataLoader.filter_trading_hours(data)
    
    # Список стратегий для сравнения
    strategies = {
        'Long Pullback': LongPullbackStrategy(),
        'Short Pullback': ShortPullbackStrategy(),
        'Neutral Range': NeutralRangeStrategy(),
        'Combined': CombinedStrategy()
    }
    
    results_summary = []
    
    # Тестируем каждую стратегию
    for name, strategy in strategies.items():
        print(f"\n{'='*70}")
        print(f"Тестирование: {name}")
        print('='*70)
        
        backtester = Backtester(strategy=strategy, data=data.copy())
        results = backtester.run()
        
        # Собираем результаты в таблицу
        results_summary.append({
            'Стратегия': name,
            'Сделок': results['total_trades'],
            'Win Rate %': f"{results['win_rate']:.1f}",
            'PnL': f"{results['total_pnl']:.0f}",
            'Profit Factor': f"{results['profit_factor']:.2f}",
            'Sharpe': f"{results['sharpe_ratio']:.2f}",
            'Max DD %': f"{results['max_drawdown']:.1f}"
        })
        
        # Сохраняем графики для каждой стратегии
        instrument_name = selected_file.replace('candles_', '').replace('.csv', '')
        backtester.plot_results(
            save_path=f'backtest_{instrument_name}_{name.replace(" ", "_").lower()}.png'
        )
    
    # Выводим сводную таблицу
    print("\n" + "="*70)
    print("📊 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*70)
    
    import pandas as pd
    summary_df = pd.DataFrame(results_summary)
    print("\n" + summary_df.to_string(index=False))
    
    # Сохраняем сравнение в CSV
    comparison_file = f'strategy_comparison_{selected_file.replace("candles_", "").replace(".csv", "")}.csv'
    summary_df.to_csv(comparison_file, index=False)
    print(f"\n💾 Сравнение сохранено: {comparison_file}")


def download_data_menu():
    """
    Меню для загрузки исторических данных через Tinkoff API
    """
    print("="*70)
    print("📥 ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ")
    print("="*70)
    
    print("\n💡 Для загрузки данных запустите:")
    print("   python download_history.py")
    
    print("\nЭтот скрипт:")
    print("   - Подключится к Tinkoff Invest API")
    print("   - Загрузит минутные свечи за нужный период")
    print("   - Сохранит в CSV файл для бэктестинга")
    
    print("\nВам понадобится:")
    print("   - Токен Tinkoff API (получить: tinkoff.ru/invest/settings/api/)")
    print("   - Тикер инструмента (например: SBER, GAZP, Si, RTS)")
    
    run_now = input("\nЗапустить загрузку сейчас? (y/n): ").strip().lower()
    
    if run_now == 'y':
        os.system('python download_history.py')

def train_ml_model_menu():
    """
    Меню для обучения ML-модели прямо из main.py
    Вызывает логику из train_ml_model.py
    """
    print("="*70)
    print("🎓 ОБУЧЕНИЕ ML МОДЕЛИ ДЛЯ ТОРГОВОГО БОТА")
    print("="*70)

    # Шаг 1: Поиск файлов с данными
    print("\n📂 Поиск файлов с историческими данными...")
    csv_files = glob.glob("candles_*.csv")
    csv_files.sort()
    
    if not csv_files:
        print("   ❌ Не найдено ни одного файла с паттерном 'candles_*.csv'")
        print("   Запустите сначала: python download_history.py")
        return
    
    print(f"\n✅ Найдено {len(csv_files)} файл(ов):\n")
    for i, file in enumerate(csv_files, 1):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"   {i}. {file} ({size_mb:.2f} MB)")
    
    # Шаг 2: Выбор файла
    while True:
        choice = input(f"\nВыберите файл (1-{len(csv_files)}, по умолчанию 1): ").strip()
        
        if choice == "" or choice == "1":
            filename = csv_files[0]
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(csv_files):
            filename = csv_files[int(choice) - 1]
            break
        else:
            print(f"   ⚠️ Введите число от 1 до {len(csv_files)}")
    
    # Шаг 3: Загрузка данных
    print(f"\n📥 Загрузка данных из {filename}...")
    try:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"   ✅ Загружено {len(df)} свечей")
        print(f"   📅 Период: {df.index[0]} - {df.index[-1]}")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки файла: {e}")
        return

    # Шаг 4: Настройки обучения
    print("\n⚙️ НАСТРОЙКИ ОБУЧЕНИЯ")
    print("-" * 70)

    forward_input = input("Горизонт прогноза (свечей вперёд, по умолчанию 5): ").strip()
    forward_periods = int(forward_input) if forward_input else 5

    test_input = input("Доля тестовых данных (по умолчанию 0.2 = 20%): ").strip()
    test_size = float(test_input) if test_input else 0.2

    print(f"\n   Горизонт прогноза: {forward_periods} свечей")
    print(f"   Тестовая выборка: {test_size*100:.0f}%")

    # Шаг 5: Создание и обучение модели
    predictor = MLPredictor()
    
    try:
        predictor.train(df, forward_periods=forward_periods, test_size=test_size)
    except KeyboardInterrupt:
        print("\n\n⚠️ Обучение прервано пользователем (Ctrl+C)")
        return
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        return

    # Шаг 6: Сохранение модели
    model_name = filename.replace('.csv', '_ml_model.pkl')
    predictor.save_model(model_name)

    # Шаг 7: Тестовый прогноз
    print("\n" + "="*70)
    print("🔮 ТЕСТОВЫЙ ПРОГНОЗ (последняя свеча)")
    print("="*70)

    prediction = predictor.predict(df)
    print(f"   Направление: {prediction['direction']}")
    print(f"   Вероятность: {prediction['probability']*100:.1f}%")
    print(f"   Уверенность: {prediction['confidence']}")
    print(f"   Prob(UP):    {prediction['prob_up']*100:.1f}%")
    print(f"   Prob(DOWN):  {prediction['prob_down']*100:.1f}%")

    print("\n" + "="*70)
    print("✅ ГОТОВО! Модель можно использовать в боте")
    print("="*70)
    print(f"\n💡 Для использования в main.py:")
    print(f"   1. Выберите тот же файл: {filename}")
    print(f"   2. Выберите стратегию: ML-Enhanced Breakout (опция 6)")
    print(f"   3. Модель автоматически загрузится из {model_name}")
    print()


# ============================================================================
# ТОЧКА ВХОДА В ПРОГРАММУ
# ============================================================================
if __name__ == "__main__":
    """
    Главная точка входа - показывает меню и обрабатывает выбор пользователя
    """
    
    try:
        print("\n" + "="*70)
        print("🤖 ТОРГОВЫЙ БОТ - БЭКТЕСТИНГ НА РЕАЛЬНЫХ ДАННЫХ")
        print("="*70)
        
        # Проверяем наличие файлов с данными
        csv_files = glob.glob("candles_*.csv")
        
        if csv_files:
            print(f"\n✅ Найдено файлов с данными: {len(csv_files)}")
        else:
            print("\n⚠️ Файлы с данными не найдены!")
            print("\n📥 Сначала нужно скачать исторические данные")
        
        # Показываем меню
        print("\n" + "-"*70)
        print("ВЫБЕРИТЕ ДЕЙСТВИЕ:")
        print("-"*70)
        print("1. Запустить бэктестинг (если есть данные)")
        print("2. Сравнить все стратегии")
        print("3. Скачать исторические данные")
        print("4. Обучить ML на данных")
        print("0. Выход")
        
        choice = input("\nВаш выбор (0-3): ").strip()
        
        # Обработка выбора пользователя
        if choice == "1":
            run_backtest_with_real_data()
        elif choice == "2":
            compare_strategies_on_real_data()
        elif choice == "3":
            download_data_menu()
        elif choice == "4":
            train_ml_model_menu()
        elif choice == "0":
            print("\n👋 До свидания!")
            sys.exit(0)
        else:
            print("\n❌ Неверный выбор")
            print("💡 Запустите бэктестинг с помощью: python main.py")
        
        print("\n✅ Программа завершена.")
    
    # ========================================================================
    # ГЛОБАЛЬНАЯ ОБРАБОТКА CTRL+C НА УРОВНЕ ПРОГРАММЫ
    # ========================================================================
    except KeyboardInterrupt:
        print("\n\n⚠️ Программа прервана пользователем (Ctrl+C)")
        print("👋 До свидания!")
        sys.exit(0)
    
    # ========================================================================
    # ОБРАБОТКА КРИТИЧЕСКИХ ОШИБОК
    # ========================================================================
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





# """
# main.py - Главный файл для запуска торгового бота
# Работает с реальными историческими данными
# """

# from datetime import datetime
# import sys
# import os

# from config import config
# from data_loader import DataLoader
# from strategies import LongPullbackStrategy, ShortPullbackStrategy, NeutralRangeStrategy, CombinedStrategy
# from backtester import Backtester
# from trading_engine import TradingEngine
# from strategies_futures import BreakoutStrategy
# from strategies_ml import MLEnhancedBreakoutStrategy
# # from strategies_futures import BreakoutStrategy, VolatilityBreakoutStrategy, CombinedFuturesStrategy

# def run_backtest_with_real_data():
#     """
#     Запуск бэктестинга на РЕАЛЬНЫХ исторических данных
#     """
#     print("="*70)
#     print("БЭКТЕСТИНГ НА РЕАЛЬНЫХ ДАННЫХ")
#     print("="*70)

#     # 1. Проверяем наличие файлов с данными
#     print("\n1️⃣ ПОИСК ФАЙЛОВ С ДАННЫМИ")
#     print("-" * 70)

#     # Ищем CSV файлы в текущей директории
#     csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and f.startswith('candles_')]

#     if csv_files:
#         print(f"\n✅ Найдено {len(csv_files)} файл(ов) с данными:\n")
#         for i, file in enumerate(csv_files, 1):
#             size_mb = os.path.getsize(file) / (1024 * 1024)
#             print(f"   {i}. {file} ({size_mb:.2f} MB)")

#         print("\n" + "-" * 70)
#         choice = input(f"\nВыберите файл (1-{len(csv_files)}) или нажмите Enter для первого: ").strip()

#         if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
#             selected_file = csv_files[int(choice) - 1]
#         else:
#             selected_file = csv_files[0]

#         print(f"\n✅ Выбран файл: {selected_file}")

#     else:
#         print("\n⚠️  Файлы с данными не найдены!")
#         print("\n💡 Сначала скачайте данные:")
#         print("   python download_history.py")
#         print("\nИли создайте CSV файл со свечами (колонки: timestamp, open, high, low, close, volume)")

#         use_test = input("\nИспользовать тестовые данные? (y/n): ").strip().lower()
#         if use_test == 'y':
#             print("\n⚠️  Используются синтетические данные для демонстрации")
#             data = DataLoader.generate_sample_data(days=30, interval_minutes=1, start_price=100000.0)
#             data = DataLoader.filter_trading_hours(data, start_time="10:00", end_time="18:45")
#             selected_file = "synthetic_data"
#         else:
#             print("\n❌ Бэктестинг отменён")
#             return None

#     # 2. Загружаем данные
#     if selected_file != "synthetic_data":
#         print("\n2️⃣ ЗАГРУЗКА ДАННЫХ")
#         print("-" * 70)

#         data = DataLoader.load_from_csv(selected_file, date_column='timestamp')

#         # Фильтруем только торговые часы
#         print("\n🕐 Фильтрация торговых часов (10:00-18:45)...")
#         data = DataLoader.filter_trading_hours(data, start_time="10:00", end_time="18:45")

#     # 3. Выбор стратегии
#     print("\nДоступные стратегии:")
#     print("1. Long Pullback - Только лонг на откатах в восходящем тренде")
#     print("2. Short Pullback - Только шорт на откатах в нисходящем тренде")
#     print("3. Neutral Range - Диапазонная торговля в боковике")
#     print("4. Combined - Все три стратегии (автоматический выбор по контексту) ⭐")
#     print("5. Breakout - Пробой диапазона (ДЛЯ ФЬЮЧЕРСОВ) 🚀")
#     print("6. Volatility Breakout - Пробой с фильтром волатильности (ДЛЯ ФЬЮЧЕРСОВ) ⭐⭐")
#     print("7. ML-Enhanced Breakout - С машинным обучением ⭐⭐⭐")

#     strategy_choice = input("\nВыберите стратегию (1-6, по умолчанию 6): ").strip()

#     if strategy_choice == "1":
#         strategy = LongPullbackStrategy()
#         strategy_name = "Long Pullback"
#     elif strategy_choice == "2":
#         strategy = ShortPullbackStrategy()
#         strategy_name = "Short Pullback"
#     elif strategy_choice == "3":
#         strategy = NeutralRangeStrategy()
#         strategy_name = "Neutral Range"
#     elif strategy_choice == "4":
#         strategy = CombinedStrategy()
#         strategy_name = "Combined"
#     elif strategy_choice == "5":
#         strategy = BreakoutStrategy()
#         strategy_name = "Breakout"
#     elif strategy_choice == "7":
#         # ✅ АВТОМАТИЧЕСКИ НАХОДИМ ML-МОДЕЛЬ ДЛЯ ВЫБРАННОГО ФАЙЛА
#         ml_model_path = selected_file.replace('.csv', '_ml_model.pkl')
        
#         if not os.path.exists(ml_model_path):
#             print(f"\n   ⚠️ ML-модель не найдена: {ml_model_path}")
#             print(f"   Создайте модель командой: python train_ml_model.py")
#             print(f"   и выберите файл: {selected_file}")
#             return None
        
#         strategy = MLEnhancedBreakoutStrategy(
#             ml_model_path=ml_model_path,
#             use_ml=True,
#             min_confidence='MEDIUM'
#         )
#         strategy_name = "ML-Enhanced Breakout"
#     else:  # По умолчанию стратегия 6
#         strategy = BreakoutStrategy()  # Простой Breakout по умолчанию
#         strategy_name = "Breakout"
    

#     # strategy_choice = input("\nВыберите стратегию (1-4, по умолчанию 4): ").strip()

#     # strategies_map = {
#     #     '1': ('Long Pullback', LongPullbackStrategy()),
#     #     '2': ('Short Pullback', ShortPullbackStrategy()),
#     #     '3': ('Neutral Range', NeutralRangeStrategy()),
#     #     '4': ('Combined', CombinedStrategy()),
#     # }

#     # strategy_name, strategy = strategies_map.get(strategy_choice, strategies_map['4'])
#     print(f"\n✅ Выбрана стратегия: {strategy_name}")

#     # 4. Запуск бэктестинга
#     print("\n4️⃣ ЗАПУСК БЭКТЕСТИНГА")
#     print("-" * 70)

#     backtester = Backtester(strategy=strategy, data=data)
#     results = backtester.run()

#     # 5. Визуализация
#     print("\n5️⃣ ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
#     print("-" * 70)
#     print("⚠️ Для остановки нажмите Ctrl+C\n")

#     backtester = Backtester(strategy=strategy, data=data)
    
#     try:
#         # Запуск бэктестинга
#         results = backtester.run()

#         instrument_name = selected_file.replace('candles_', '').replace('.csv', '')
#         result_filename = f'backtest_{instrument_name}_{strategy_name.replace(" ", "_").lower()}.png'
#         backtester.plot_results(save_path=result_filename)
        
#         # 6. Сохранение лога сделок
#         trade_log = backtester.get_trade_log()
#         if not trade_log.empty:
#             log_filename = f'trades_{instrument_name}_{strategy_name.replace(" ", "_").lower()}.csv'
#             trade_log.to_csv(log_filename, index=False)
#             print(f"📄 Детальный лог сделок: {log_filename}")
            
#             print("\n📊 ПЕРВЫЕ 10 СДЕЛОК:")
#             print("-" * 70)
#             print(trade_log.head(10).to_string(index=False))

#     # # Имя файла с результатами
#     # instrument_name = selected_file.replace('candles_', '').replace('.csv', '')
#     # result_filename = f'backtest_{instrument_name}_{strategy_name.replace(" ", "_").lower()}.png'

#     # backtester.plot_results(save_path=result_filename)

#     # # 6. Сохранение лога сделок
#     # trade_log = backtester.get_trade_log()
#     # if not trade_log.empty:
#     #     log_filename = f'trades_{instrument_name}_{strategy_name.replace(" ", "_").lower()}.csv'
#     #     trade_log.to_csv(log_filename, index=False)
#     #     print(f"📄 Детальный лог сделок: {log_filename}")

#     #     print("\n📊 ПЕРВЫЕ 10 СДЕЛОК:")
#     #     print("-" * 70)
#     #     print(trade_log.head(10).to_string(index=False))

#     # 7. Итоговая статистика
#     print("\n" + "="*70)
#     print("📈 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
#     print("="*70)

#     if results['total_trades'] > 0:
#         print(f"\n✅ Стратегия: {strategy_name}")
#         print(f"📁 Данные: {selected_file}")
#         print(f"📊 График: {result_filename}")
#         print(f"📄 Лог: {log_filename if not trade_log.empty else 'N/A'}")

#         # Оценка результатов
#         print("\n" + "-"*70)
#         print("ОЦЕНКА РЕЗУЛЬТАТОВ:")
#         print("-"*70)

#         if results['win_rate'] >= 55:
#             print("✅ Win Rate отличный (≥55%)")
#         elif results['win_rate'] >= 45:
#             print("⚠️  Win Rate средний (45-55%)")
#         else:
#             print("❌ Win Rate низкий (<45%)")

#         if results['profit_factor'] >= 1.5:
#             print("✅ Profit Factor отличный (≥1.5)")
#         elif results['profit_factor'] >= 1.0:
#             print("⚠️  Profit Factor средний (1.0-1.5)")
#         else:
#             print("❌ Profit Factor низкий (<1.0)")

#         if results['sharpe_ratio'] >= 1.0:
#             print("✅ Sharpe Ratio хороший (≥1.0)")
#         elif results['sharpe_ratio'] >= 0.5:
#             print("⚠️  Sharpe Ratio средний (0.5-1.0)")
#         else:
#             print("❌ Sharpe Ratio низкий (<0.5)")

#         if results['max_drawdown'] <= 15:
#             print("✅ Просадка приемлемая (≤15%)")
#         elif results['max_drawdown'] <= 25:
#             print("⚠️  Просадка высокая (15-25%)")
#         else:
#             print("❌ Просадка критическая (>25%)")

#         # Рекомендации
#         print("\n" + "-"*70)
#         print("РЕКОМЕНДАЦИИ:")
#         print("-"*70)

#         if results['total_pnl'] > 0:
#             print("✅ Стратегия прибыльна на исторических данных")
#             print("\n💡 Следующие шаги:")
#             print("   1. Протестируйте на других периодах времени")
#             print("   2. Оптимизируйте параметры в config.py")
#             print("   3. Проведите walk-forward анализ")
#             print("   4. Запустите на forward testing (out-of-sample)")
#         else:
#             print("❌ Стратегия убыточна на этих данных")
#             print("\n💡 Попробуйте:")
#             print("   1. Изменить параметры в config.py:")
#             print("      - STOP_LOSS_ATR_MULTIPLIER (сейчас 2.0)")
#             print("      - TAKE_PROFIT_ATR_MULTIPLIER (сейчас 3.5)")
#             print("      - COOLDOWN_MINUTES (сейчас 15)")
#             print("   2. Протестировать другую стратегию")
#             print("   3. Использовать другой инструмент")
#     else:
#         print("\n⚠️  Сделок не было!")
#         print("\nВозможные причины:")
#         print("   1. Недостаточно данных для расчёта индикаторов")
#         print("   2. Контекст рынка не подходит для выбранной стратегии")
#         print("   3. Параметры входа слишком строгие")
#         print("\n💡 Попробуйте Combined стратегию или другой период данных")

#     print("\n" + "="*70)

#     return results, backtester


# def compare_strategies_on_real_data():
#     """
#     Сравнивает все стратегии на одних и тех же данных
#     """
#     print("="*70)
#     print("СРАВНЕНИЕ ВСЕХ СТРАТЕГИЙ")
#     print("="*70)

#     # Загружаем данные
#     csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and f.startswith('candles_')]

#     if not csv_files:
#         print("\n❌ Файлы с данными не найдены!")
#         print("   Запустите: python download_history.py")
#         return

#     selected_file = csv_files[0]
#     print(f"\n📂 Используется: {selected_file}")

#     data = DataLoader.load_from_csv(selected_file, date_column='timestamp')
#     data = DataLoader.filter_trading_hours(data)

#     strategies = {
#         'Long Pullback': LongPullbackStrategy(),
#         'Short Pullback': ShortPullbackStrategy(),
#         'Neutral Range': NeutralRangeStrategy(),
#         'Combined': CombinedStrategy()
#     }

#     results_summary = []

#     for name, strategy in strategies.items():
#         print(f"\n{'='*70}")
#         print(f"Тестирование: {name}")
#         print('='*70)

#         backtester = Backtester(strategy=strategy, data=data.copy())
#         results = backtester.run()

#         results_summary.append({
#             'Стратегия': name,
#             'Сделок': results['total_trades'],
#             'Win Rate %': f"{results['win_rate']:.1f}",
#             'PnL': f"{results['total_pnl']:.0f}",
#             'Profit Factor': f"{results['profit_factor']:.2f}",
#             'Sharpe': f"{results['sharpe_ratio']:.2f}",
#             'Max DD %': f"{results['max_drawdown']:.1f}"
#         })

#         # Сохраняем графики
#         instrument_name = selected_file.replace('candles_', '').replace('.csv', '')
#         backtester.plot_results(
#             save_path=f'backtest_{instrument_name}_{name.replace(" ", "_").lower()}.png'
#         )

#     # Сводная таблица
#     print("\n" + "="*70)
#     print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
#     print("="*70)

#     import pandas as pd
#     summary_df = pd.DataFrame(results_summary)
#     print("\n" + summary_df.to_string(index=False))

#     comparison_file = f'strategy_comparison_{selected_file.replace("candles_", "").replace(".csv", "")}.csv'
#     summary_df.to_csv(comparison_file, index=False)
#     print(f"\n💾 Сравнение сохранено: {comparison_file}")


# def download_data_menu():
#     """
#     Меню для загрузки данных
#     """
#     print("="*70)
#     print("ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ")
#     print("="*70)
#     print("\n💡 Для загрузки данных запустите:")
#     print("   python download_history.py")
#     print("\nЭтот скрипт:")
#     print("   - Подключится к Tinkoff Invest API")
#     print("   - Загрузит минутные свечи за нужный период")
#     print("   - Сохранит в CSV файл для бэктестинга")
#     print("\nВам понадобится:")
#     print("   - Токен Tinkoff API (получить: tinkoff.ru/invest/settings/api/)")
#     print("   - Тикер инструмента (например: SBER, GAZP, Si, RTS)")

#     run_now = input("\nЗапустить загрузку сейчас? (y/n): ").strip().lower()
#     if run_now == 'y':
#         os.system('python download_history.py')


# if __name__ == "__main__":
#     """
#     Точка входа в программу
#     """
#     print("\n" + "="*70)
#     print("🤖 ТОРГОВЫЙ БОТ - БЭКТЕСТИНГ НА РЕАЛЬНЫХ ДАННЫХ")
#     print("="*70)

#     # Проверяем наличие данных
#     csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and f.startswith('candles_')]

#     if csv_files:
#         print(f"\n✅ Найдено файлов с данными: {len(csv_files)}")
#     else:
#         print("\n⚠️  Файлы с данными не найдены!")
#         print("\n📥 Сначала нужно скачать исторические данные")

#     # Меню
#     print("\n" + "-"*70)
#     print("ВЫБЕРИТЕ ДЕЙСТВИЕ:")
#     print("-"*70)
#     print("1. Запустить бэктестинг (если есть данные)")
#     print("2. Сравнить все стратегии")
#     print("3. Скачать исторические данные")
#     print("0. Выход")

#     choice = input("\nВаш выбор (0-3): ").strip()

#     if choice == "1":
#         run_backtest_with_real_data()
#     elif choice == "2":
#         compare_strategies_on_real_data()
#     elif choice == "3":
#         download_data_menu()
#     elif choice == "0":
#         print("\n👋 До свидания!")
#         sys.exit(0)
#     else:
#         print("\n❌ Неверный выбор")
#         print("💡 Запустите бэктестинг с помощью: python main.py")

#     print("\n✅ Программа завершена.")
