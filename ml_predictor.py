"""
ml_predictor.py - ML фильтр для улучшения торговых сигналов
Использует Random Forest для прогнозирования направления цены
Оптимизирован для фьючерсов Si/RTS
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')


class MLPredictor:
    """
    ML модель для фильтрации торговых сигналов

    Прогнозирует: UP (цена вырастет) или DOWN (цена упадёт)
    Используется как дополнительный фильтр к Breakout стратегии
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.accuracy = 0.0

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создаёт 30+ фичи для ML модели

        Категории:
        1. Скользящие средние (SMA, EMA)
        2. Технические индикаторы (RSI, MACD, Bollinger Bands)
        3. Волатильность (ATR)
        4. Объём
        5. Momentum
        6. Паттерны свечей
        """
        features = pd.DataFrame(index=df.index)

        # ==========================================
        # 1. СКОЛЬЗЯЩИЕ СРЕДНИЕ
        # ==========================================
        for period in [5, 10, 20, 50, 100]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']

            # Экспоненциальная скользящая
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # ==========================================
        # 2. RSI (Relative Strength Index)
        # ==========================================
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # RSI разных периодов
        for period in [7, 21]:
            gain_p = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss_p = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs_p = gain_p / loss_p
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs_p))

        # ==========================================
        # 3. MACD (Moving Average Convergence Divergence)
        # ==========================================
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # ==========================================
        # 4. BOLLINGER BANDS
        # ==========================================
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        features['bb_middle'] = sma_20
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

        # ==========================================
        # 5. ATR (Average True Range) - Волатильность
        # ==========================================
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features['atr_14'] = true_range.rolling(14).mean()
        features['atr_normalized'] = features['atr_14'] / df['close']

        # ==========================================
        # 6. ОБЪЁМ
        # ==========================================
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        features['volume_change'] = df['volume'].pct_change()

        # ==========================================
        # 7. MOMENTUM (Импульс)
        # ==========================================
        for period in [3, 5, 10, 20]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)
            features[f'momentum_high_{period}'] = df['high'].pct_change(period)
            features[f'momentum_low_{period}'] = df['low'].pct_change(period)

        # ==========================================
        # 8. СВЕЧНЫЕ ПАТТЕРНЫ
        # ==========================================
        # Размер тела свечи
        features['candle_body'] = np.abs(df['close'] - df['open']) / df['open']

        # Верхняя/нижняя тень
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']

        # Бычья/медвежья свеча
        features['is_bullish'] = (df['close'] > df['open']).astype(int)

        # ==========================================
        # 9. ЛАГИ (Прошлые значения)
        # ==========================================
        for lag in [1, 2, 3, 5, 10]:
            features[f'return_lag_{lag}'] = df['close'].pct_change().shift(lag)
            features[f'volume_lag_{lag}'] = df['volume'].pct_change().shift(lag)

        # ==========================================
        # 10. ВРЕМЯ ДНЯ (для внутридневной торговли)
        # ==========================================
        if hasattr(df.index, 'hour'):
            features['hour'] = df.index.hour
            features['minute'] = df.index.minute
            # Время как синусоида (циклическая фича)
            features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

        # ==========================================
        # ПОСТОБРАБОТКА
        # ==========================================
        # Заполняем NaN
        features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)

        # Заменяем inf на большие числа
        features.replace([np.inf, -np.inf], [999, -999], inplace=True)

        self.feature_names = features.columns.tolist()
        return features

    def prepare_training_data(self, df: pd.DataFrame, forward_periods: int = 5):
        """
        Подготавливает данные для обучения

        Args:
            df: Исторические данные
            forward_periods: Через сколько свечей проверять результат

        Returns:
            X (features), y (labels: 1=UP, 0=DOWN)
        """
        print(f"📊 Подготовка данных для обучения...")

        # Создаём фичи
        X = self.create_features(df)

        # Создаём таргет: цена вырастет или упадёт через N свечей?
        future_returns = df['close'].shift(-forward_periods) / df['close'] - 1

        # Классификация: UP (1) если доходность > 0, иначе DOWN (0)
        y = (future_returns > 0).astype(int)

        # Удаляем последние N строк (нет будущих данных)
        X = X.iloc[:-forward_periods]
        y = y.iloc[:-forward_periods]

        # Удаляем строки с NaN в таргете
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"   Всего сэмплов: {len(X)}")
        print(f"   Фичей: {X.shape[1]}")
        print(f"   UP: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"   DOWN: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")

        return X, y

    def train(self, df: pd.DataFrame, forward_periods: int = 5, test_size: float = 0.2):
        """
        Обучает модель на исторических данных с валидацией

        Args:
            df: DataFrame с историческими данными
            forward_periods: Горизонт прогноза (свечей вперёд)
            test_size: Доля тестовых данных (0.2 = 20%)
        """
        print("\n" + "="*70)
        print("🤖 ОБУЧЕНИЕ ML МОДЕЛИ")
        print("="*70)

        # Подготовка данных
        X, y = self.prepare_training_data(df, forward_periods)

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Не перемешиваем временные ряды!
        )

        print(f"\n📈 Обучающая выборка: {len(X_train)} сэмплов")
        print(f"📉 Тестовая выборка: {len(X_test)} сэмплов")

        # Нормализация
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Обучение Random Forest
        print("\n🌳 Обучение Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=200,        # Больше деревьев = лучше, но медленнее
            max_depth=15,            # Ограничение глубины против переобучения
            min_samples_split=50,    # Минимум сэмплов для разбиения
            min_samples_leaf=20,     # Минимум сэмплов в листе
            max_features='sqrt',     # Случайный выбор фичей
            random_state=42,
            n_jobs=-1,               # Все ядра CPU
            class_weight='balanced'  # Балансировка классов
        )

        self.model.fit(X_train_scaled, y_train)
        if train_accuracy - test_accuracy > 0.15:  # Разница >15%
            print("⚠️ ПЕРЕОБУЧЕНИЕ! Модель слишком подогнана под обучающие данные")
            print(f"   Train: {train_accuracy*100:.1f}% vs Test: {test_accuracy*100:.1f}%")
            print("   Рекомендация: уменьшите max_depth или увеличьте min_samples_split")
        self.is_trained = True

        # Оценка на обучающей выборке
        y_train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Оценка на тестовой выборке (ВАЖНО!)
        y_test_pred = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        self.accuracy = test_accuracy

        print("\n" + "="*70)
        print("✅ РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        print("="*70)
        print(f"Точность на обучающих данных: {train_accuracy*100:.1f}%")
        print(f"Точность на ТЕСТОВЫХ данных:  {test_accuracy*100:.1f}% ⭐")
        print()

        # Детальный отчёт
        print("📊 Детальная статистика (тестовая выборка):")
        print(classification_report(y_test, y_test_pred, 
                                   target_names=['DOWN', 'UP'],
                                   digits=3))

        # Feature importance
        print("\n🔍 Топ-10 важных признаков:")
        importances = self.model.feature_importances_
        top_features = sorted(zip(self.feature_names, importances), 
                            key=lambda x: x[1], reverse=True)[:10]
        for i, (feat, imp) in enumerate(top_features, 1):
            print(f"   {i}. {feat:30s}: {imp:.4f}")

        # Рекомендации
        print("\n" + "="*70)
        if test_accuracy >= 0.55:
            print("✅ Модель готова к использованию!")
            print("   Точность >55% - это хорошо для финансовых рынков")
        elif test_accuracy >= 0.52:
            print("⚠️ Модель можно использовать с осторожностью")
            print("   Точность 52-55% - слабый сигнал")
        else:
            print("❌ Модель не лучше случайного угадывания")
            print("   Рекомендуется:")
            print("   1. Загрузить больше данных (30+ дней)")
            print("   2. Изменить forward_periods")
            print("   3. Добавить дополнительные фичи")
        print("="*70)

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Делает прогноз на основе последних данных

        Returns:
            {
                'direction': 'UP' или 'DOWN',
                'probability': вероятность (0-1),
                'confidence': 'HIGH', 'MEDIUM', 'LOW'
            }
        """
        if not self.is_trained:
            return {
                'direction': 'NEUTRAL', 
                'probability': 0.5, 
                'confidence': 'NONE'
            }

        # Создаём фичи для последней свечи
        X = self.create_features(df)
        X_last = X.iloc[[-1]]

        # Нормализация
        X_scaled = self.scaler.transform(X_last)

        # Прогноз
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Вероятность для выбранного класса
        prob = probabilities[prediction]

        # Уровень уверенности
        if prob >= 0.70:
            confidence = 'HIGH'
        elif prob >= 0.60:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        return {
            'direction': 'UP' if prediction == 1 else 'DOWN',
            'probability': float(prob),
            'confidence': confidence,
            'prob_up': float(probabilities[1]),
            'prob_down': float(probabilities[0])
        }

    def save_model(self, filepath: str = 'ml_model.pkl'):
        """Сохраняет обученную модель"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'accuracy': self.accuracy
            }, f)
        print(f"\n💾 Модель сохранена: {filepath}")

    def load_model(self, filepath: str = 'ml_model.pkl'):
        """Загружает обученную модель"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.accuracy = data.get('accuracy', 0.0)
            self.is_trained = True
        print(f"📂 Модель загружена: {filepath}")
        print(f"   Точность: {self.accuracy*100:.1f}%")
