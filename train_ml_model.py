"""
train_ml_model.py - Скрипт для обучения ML модели
Запустите этот скрипт перед использованием ML в боте
"""

import pandas as pd
from ml_predictor import MLPredictor

def main():
    print("="*70)
    print("🎓 ОБУЧЕНИЕ ML МОДЕЛИ ДЛЯ ТОРГОВОГО БОТА")
    print("="*70)

    # 1. Выбор файла с данными
    print("\n📂 Доступные файлы:")
    print("   1. candles_SiZ5_15days.csv")
    print("   2. candles_SBER_30days.csv")
    print("   3. Другой файл (введите имя)")

    choice = input("\nВыберите файл (1-3, по умолчанию 1): ").strip()

    if choice == "2":
        filename = "candles_SBER_30days.csv"
    elif choice == "3":
        filename = input("Введите имя файла: ").strip()
    else:
        filename = "candles_SiZ5_15days.csv"

    # 2. Загрузка данных
    print(f"\n📥 Загрузка данных из {filename}...")
    try:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"   ✅ Загружено {len(df)} свечей")
        print(f"   📅 Период: {df.index[0]} - {df.index[-1]}")
    except FileNotFoundError:
        print(f"   ❌ Файл {filename} не найден!")
        print("   Запустите сначала: python download_history.py")
        return

    # 3. Настройки обучения
    print("\n⚙️ НАСТРОЙКИ ОБУЧЕНИЯ")
    print("-" * 70)

    forward_input = input("Горизонт прогноза (свечей вперёд, по умолчанию 5): ").strip()
    forward_periods = int(forward_input) if forward_input else 5

    test_input = input("Доля тестовых данных (по умолчанию 0.2 = 20%): ").strip()
    test_size = float(test_input) if test_input else 0.2

    print(f"\n   Горизонт прогноза: {forward_periods} свечей")
    print(f"   Тестовая выборка: {test_size*100:.0f}%")

    # 4. Создание и обучение модели
    predictor = MLPredictor()
    predictor.train(df, forward_periods=forward_periods, test_size=test_size)

    # 5. Сохранение модели
    model_name = filename.replace('.csv', '_ml_model.pkl')
    predictor.save_model(model_name)

    # 6. Тестовый прогноз
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
