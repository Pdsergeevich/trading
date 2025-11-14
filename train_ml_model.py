"""
train_ml_model.py - Скрипт для обучения ML модели
Запустите этот скрипт перед использованием ML в боте
"""

import pandas as pd
import glob
import os
from ml_predictor import MLPredictor


def main():
    print("="*70)
    print("🎓 ОБУЧЕНИЕ ML МОДЕЛИ ДЛЯ ТОРГОВОГО БОТА")
    print("="*70)

    # 1. Автоматический поиск всех файлов с данными
    print("\n📂 Поиск файлов с историческими данными...")
    
    # Ищем все CSV файлы, начинающиеся с "candles_"
    csv_files = glob.glob("candles_*.csv")
    
    if not csv_files:
        print("   ❌ Не найдено ни одного файла с паттерном 'candles_*.csv'")
        print("   Запустите сначала: python download_history.py")
        return
    
    # Сортируем файлы по имени
    csv_files.sort()
    
    print(f"\n📂 Найдено файлов: {len(csv_files)}")
    print("\nДоступные файлы:")
    for i, file in enumerate(csv_files, 1):
        # Показываем размер файла
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"   {i}. {file} ({size_mb:.2f} MB)")
    
    # 2. Выбор файла
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
    
    # 3. Загрузка данных
    print(f"\n📥 Загрузка данных из {filename}...")
    try:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"   ✅ Загружено {len(df)} свечей")
        print(f"   📅 Период: {df.index[0]} - {df.index[-1]}")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки файла: {e}")
        return

    # 4. Настройки обучения
    print("\n⚙️ НАСТРОЙКИ ОБУЧЕНИЯ")
    print("-" * 70)

    forward_input = input("Горизонт прогноза (свечей вперёд, по умолчанию 5): ").strip()
    forward_periods = int(forward_input) if forward_input else 5

    test_input = input("Доля тестовых данных (по умолчанию 0.2 = 20%): ").strip()
    test_size = float(test_input) if test_input else 0.2

    print(f"\n   Горизонт прогноза: {forward_periods} свечей")
    print(f"   Тестовая выборка: {test_size*100:.0f}%")

    # 5. Создание и обучение модели
    predictor = MLPredictor()
    predictor.train(df, forward_periods=forward_periods, test_size=test_size)

    # 6. Сохранение модели
    # Сохраняем с полным именем, включая количество дней
    model_name = filename.replace('.csv', '_ml_model.pkl')
    predictor.save_model(model_name)

    print(f"\n✅ Модель сохранена: {model_name}")

    # Извлекаем тикер для информационного сообщения
    instrument_name = filename.replace('candles_', '').split('_')[0]

    print(f"\n💡 Использование модели:")
    print(f"   - При выборе ML-стратегии система автоматически найдёт")
    print(f"     ЛЮБУЮ модель для {instrument_name}, обученную на любом периоде")
    print(f"   - Можете обучить несколько моделей на разных периодах")
    print(f"     и выбрать лучшую при запуске бэктестинга")

    # 7. Тестовый прогноз
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
    print(f"   1. Выберите стратегию: ML-Enhanced Breakout")
    print(f"   2. Модель автоматически загрузится из {model_name}")
    print()


if __name__ == "__main__":
    main()
