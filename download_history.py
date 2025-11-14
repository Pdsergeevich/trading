"""
download_history.py - Скрипт для загрузки исторических минутных свечей
Скачивает реальные данные через Tinkoff Invest API
"""

from datetime import datetime, timedelta
import pandas as pd
import os

# ВАЖНО: Установите библиотеку: pip install tinkoff-investments

def download_tinkoff_candles(token: str, figi: str, days: int = 30, ticker_name: str = ""):
    """
    Загружает минутные свечи через Tinkoff Invest API

    Args:
        token: Токен Tinkoff Invest API
        figi: FIGI инструмента
        days: Количество дней истории
        ticker_name: Название тикера для имени файла
    """
    try:
        from tinkoff.invest import Client, CandleInterval
        from tinkoff.invest.utils import now
    except ImportError:
        print("❌ Ошибка: Установите библиотеку")
        print("   pip install tinkoff-investments")
        return None, None

    print(f"📥 Загрузка данных через Tinkoff API...")
    print(f"   FIGI: {figi}")
    print(f"   Период: {days} дней")

    all_candles = []
    instrument_name = ticker_name

    with Client(token) as client:
        # Получаем информацию об инструменте
        try:
            instrument = client.instruments.get_instrument_by(
                id_type=1,  # FIGI
                id=figi
            ).instrument
            print(f"   Инструмент: {instrument.name} ({instrument.ticker})")
            instrument_name = instrument.ticker
        except Exception as e:
            print(f"⚠️  Не удалось получить информацию об инструменте: {e}")
            print(f"   Возможно, FIGI устарел или неверен")
            return None, None

        # Tinkoff API ограничивает запрос 1 днём для минутных свечей
        # Поэтому загружаем по дням
        end_date = now()

        for day in range(days):
            day_end = end_date - timedelta(days=day)
            day_start = day_end - timedelta(days=1)

            try:
                candles = client.market_data.get_candles(
                    figi=figi,
                    from_=day_start,
                    to=day_end,
                    interval=CandleInterval.CANDLE_INTERVAL_1_MIN
                )

                for candle in candles.candles:
                    all_candles.append({
                        'timestamp': candle.time,
                        'open': _quotation_to_float(candle.open),
                        'high': _quotation_to_float(candle.high),
                        'low': _quotation_to_float(candle.low),
                        'close': _quotation_to_float(candle.close),
                        'volume': candle.volume
                    })

                print(f"   ✓ День {day+1}/{days}: загружено {len(candles.candles)} свечей")

            except Exception as e:
                print(f"   ✗ Ошибка при загрузке дня {day+1}: {e}")
                continue

    if not all_candles:
        print("❌ Не удалось загрузить данные")
        print("\n💡 Возможные причины:")
        print("   1. FIGI инструмента устарел (для фьючерсов)")
        print("   2. Инструмент не торгуется в выбранный период")
        print("   3. Проблемы с подключением к API")
        return None, None

    # Создаём DataFrame
    df = pd.DataFrame(all_candles)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)

    # Удаляем дубликаты
    df = df[~df.index.duplicated(keep='first')]

    print(f"\n✅ Загружено {len(df)} свечей")
    print(f"   Период: {df.index[0]} - {df.index[-1]}")

    return df, instrument_name

def _quotation_to_float(quotation) -> float:
    """Конвертирует Quotation в float"""
    return quotation.units + quotation.nano / 1e9


def get_popular_instruments():
    """Возвращает список популярных АКЦИЙ с FIGI"""
    return {
        # ТОЛЬКО АКЦИИ (FIGI не меняются!)
        'SBER': 'BBG004730N88',    # Сбербанк
        'GAZP': 'BBG004730RP0',    # Газпром
        'LKOH': 'BBG004731032',    # ЛУКОЙЛ
        'YNDX': 'BBG00FGZB3N3',    # Яндекс
        'GMKN': 'BBG004731489',    # ГМК Норникель
        'NVTK': 'BBG004731354',    # Новатэк
        'ROSN': 'BBG004731126',    # Роснефть
        'TATN': 'BBG004RVFCY3',    # Татнефть
        'MGNT': 'BBG004MHGR69',    # Магнит
        'MTSS': 'BBG004SV7YE9',    # МТС
        'VTBR': 'BBG004730ZJ9',    # ВТБ
        'PLZL': 'BBG000R607Y3',    # Полюс
    }


def find_active_future(token: str, base_asset: str):
    """
    Находит активный (ближайший) фьючерс по базовому активу

    Args:
        token: Токен Tinkoff API
        base_asset: Базовый актив ('Si' для USD, 'RTS' для индекса и т.д.)

    Returns:
        Tuple (figi, ticker, expiration_date) или (None, None, None)
    """
    try:
        from tinkoff.invest import Client
    except ImportError:
        return None, None, None

    print(f"🔍 Поиск активного фьючерса для {base_asset}...")

    with Client(token) as client:
        futures = client.instruments.futures()

        # Фильтруем фьючерсы по базовому активу
        matching_futures = []

        for future in futures.instruments:
            # Проверяем что тикер содержит базовый актив
            if base_asset.upper() in future.ticker.upper():
                # Проверяем что не истёк
                if future.expiration_date > datetime.now(future.expiration_date.tzinfo):
                    matching_futures.append({
                        'figi': future.figi,
                        'ticker': future.ticker,
                        'name': future.name,
                        'expiration': future.expiration_date
                    })

        if not matching_futures:
            return None, None, None

        # Сортируем по дате экспирации (ближайший первый)
        matching_futures.sort(key=lambda x: x['expiration'])

        # Берём ближайший
        nearest = matching_futures[0]

        print(f"✅ Найден активный фьючерс:")
        print(f"   Тикер: {nearest['ticker']}")
        print(f"   Название: {nearest['name']}")
        print(f"   FIGI: {nearest['figi']}")
        print(f"   Экспирация: {nearest['expiration'].strftime('%Y-%m-%d')}")

        return nearest['figi'], nearest['ticker'], nearest['expiration']

    return None, None, None


def find_instrument_by_ticker(token: str, ticker: str):
    """
    Находит инструмент по тикеру (акции или фьючерсы)

    Args:
        token: Токен Tinkoff API
        ticker: Тикер инструмента

    Returns:
        Tuple (figi, instrument_name) или (None, None)
    """
    try:
        from tinkoff.invest import Client
    except ImportError:
        print("❌ Установите: pip install tinkoff-investments")
        return None, None

    with Client(token) as client:
        # 1. Сначала ищем среди акций
        print(f"🔍 Поиск акции '{ticker}'...")
        shares = client.instruments.shares()
        for share in shares.instruments:
            if share.ticker.upper() == ticker.upper():
                print(f"✅ Найдена акция: {share.name}")
                print(f"   Тикер: {share.ticker}")
                print(f"   FIGI: {share.figi}")
                print(f"   Валюта: {share.currency}")
                return share.figi, share.ticker

        # 2. Если не нашли акцию - ищем фьючерс
        print(f"🔍 Акция не найдена, ищу фьючерс '{ticker}'...")

        # Для популярных фьючерсов
        if ticker.upper() in ['SI', 'RTS', 'GOLD', 'BR']:
            figi, future_ticker, exp_date = find_active_future(token, ticker)
            if figi:
                return figi, future_ticker

        # Общий поиск фьючерсов
        futures = client.instruments.futures()
        for future in futures.instruments:
            if ticker.upper() == future.ticker.upper():
                # Проверяем что не истёк
                if future.expiration_date > datetime.now(future.expiration_date.tzinfo):
                    print(f"✅ Найден фьючерс: {future.name}")
                    print(f"   Тикер: {future.ticker}")
                    print(f"   FIGI: {future.figi}")
                    print(f"   Экспирация: {future.expiration_date.strftime('%Y-%m-%d')}")
                    return future.figi, future.ticker

    print(f"❌ Инструмент '{ticker}' не найден")
    return None, None


if __name__ == "__main__":
    """
    Пример использования скрипта для загрузки данных
    """
    print("="*80)
    print("ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ ИЗ TINKOFF INVEST API")
    print("="*80)

    # ВАЖНО: Укажите ваш токен!
    TOKEN = input("\nВведите ваш Tinkoff API токен: ").strip()

    if not TOKEN or TOKEN == "":
        print("\n❌ Токен не указан!")
        print("\n💡 Получите токен:")
        print("   1. Откройте приложение Тinkoff Инвестиции")
        print("   2. Настройки → Токены для API")
        print("   3. Выпустите токен с правами 'Только чтение'")
        exit(1)

    print("\n" + "-"*80)
    print("ПОПУЛЯРНЫЕ АКЦИИ (рекомендуется для начала):")
    print("-"*80)

    instruments = get_popular_instruments()
    instruments_list = list(instruments.items())

    for i, (ticker, figi) in enumerate(instruments_list, 1):
        print(f"{i:2d}. {ticker:6s}")

    print("\n" + "-"*80)
    print("💡 Можно также использовать ФЬЮЧЕРСЫ (Si, RTS)")
    print("   Для фьючерсов будет найден актуальный контракт автоматически")
    print("-"*80)

    print("\n💡 Введите:")
    print("   • НОМЕР из списка (1-{})".format(len(instruments_list)))
    print("   • ТИКЕР акции (например: SBER, GAZP)")
    print("   • ТИКЕР фьючерса (например: Si, RTS)")

    choice = input("\nВаш выбор: ").strip().upper()

    # Получаем FIGI
    figi = None
    ticker_name = None

    # Проверяем, это номер или тикер
    if choice.isdigit():
        choice_num = int(choice)
        if 1 <= choice_num <= len(instruments_list):
            ticker_name, figi = instruments_list[choice_num - 1]
            print(f"\n✅ Выбрана акция: {ticker_name}")
            print(f"   FIGI: {figi}")
        else:
            print(f"\n❌ Неверный номер! Выберите от 1 до {len(instruments_list)}")
            exit(1)
    else:
        # Проверяем популярные инструменты
        if choice in instruments:
            ticker_name = choice
            figi = instruments[choice]
            print(f"\n✅ Выбрана акция: {ticker_name}")
            print(f"   FIGI: {figi}")
        else:
            # Ищем через API (акции или фьючерсы)
            figi, ticker_name = find_instrument_by_ticker(TOKEN, choice)
            if not figi:
                print("\n❌ Инструмент не найден!")
                print("\n💡 Попробуйте:")
                print("   • Выбрать акцию по номеру из списка")
                print("   • Ввести точный тикер (например: SBER)")
                exit(1)

    # Выбор периода
    print("\n" + "-"*80)
    days_input = input("Сколько дней истории загрузить? (по умолчанию 30): ").strip()
    days = int(days_input) if days_input.isdigit() and int(days_input) > 0 else 30

    # Загружаем данные
    print("\n" + "-"*80)
    df, actual_ticker = download_tinkoff_candles(TOKEN, figi, days, ticker_name)

    if df is not None and actual_ticker:
        # Сохраняем в CSV
        filename = f"candles_{actual_ticker}_{days}days.csv"
        df.to_csv(filename)
        print(f"\n💾 Данные сохранены в файл: {filename}")

        # Показываем статистику
        print("\n" + "-"*80)
        print("СТАТИСТИКА ДАННЫХ:")
        print("-"*80)
        print(f"Всего свечей: {len(df)}")
        print(f"Период: {df.index[0]} - {df.index[-1]}")
        print(f"Мин. цена: {df['low'].min():.2f}")
        print(f"Макс. цена: {df['high'].max():.2f}")
        print(f"Средний объём: {df['volume'].mean():.0f}")

        print("\nПервые 5 свечей:")
        print(df.head())

        print("\n" + "="*80)
        print("✅ ГОТОВО! Теперь запустите бэктестинг:")
        print("="*80)
        print(f"\n  python main.py")
        print(f"\n  Выберите файл: {filename}")
        print("\n" + "="*80)
    else:
        print("\n❌ Ошибка при загрузке данных")
        print("\n💡 Рекомендация: попробуйте выбрать акцию из списка (1-12)")
