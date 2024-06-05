import os
import pandas as pd
import warnings

from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from joblib import load


DATASET_PATH = os.environ.get('DATASET_PATH', '../datasets/')


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        raise RuntimeError(f'Пути {DATASET_PATH} не существует')
    
    # Загрузка датасета
    df_train = pd.read_csv(os.path.join(DATASET_PATH, 'train.csv'))
    if df_train is None or not len(df_train):
        raise RuntimeError('Тренировочный датасет не считан')

    df_test = pd.read_csv(os.path.join(DATASET_PATH, 'test.csv'))
    if df_test is None or not len(df_test):
        raise RuntimeError('Валидационный датасет не считан')
    
    # Загрузка моделей
    model: XGBRegressor = load(os.path.join(DATASET_PATH, 'xgbregressor.joblib'))
    if model is None:
        # Это проблема, серьезная проблема
        raise RuntimeError(f"Не найдена модель для xgbregressor.joblib")
    
    model_linear: SGDRegressor = load(os.path.join(DATASET_PATH, 'sgdregressor.joblib'))
    if model is None:
        # Это проблема, серьезная проблема
        raise RuntimeError(f"Не найдена модель для sgdregressor.joblib")
    
    # Подготовка данных для теста
    X_train = df_train.drop('Price(euro)', axis=1)
    y_train = df_train['Price(euro)']
    X_test = df_test.drop('Price(euro)', axis=1)
    y_test = df_test['Price(euro)']

    # Тест моделей
    r2_xgb_train = r2_score(y_train, model.predict(X_train))
    r2_xgb_test = r2_score(y_test, model.predict(X_test))
    r2_sgd_train = r2_score(y_train, model_linear.predict(X_train))
    r2_sgd_test = r2_score(y_test, model_linear.predict(X_test))

    print(f'R2 на тренировочных данных XGBRegressor: {r2_xgb_train:.2f}')
    print(f'R2 на валидационных данных XGBRegressor: {r2_xgb_test:.2f}')
    print(f'R2 на тренировочных данных SGDRegressor: {r2_sgd_train:.2f}')
    print(f'R2 на валидационных данных SGDRegressor: {r2_sgd_test:.2f}')