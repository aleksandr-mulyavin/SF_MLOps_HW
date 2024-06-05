import os
import pandas as pd
import warnings

from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from joblib import dump


DATASET_PATH = os.environ.get('DATASET_PATH', '../datasets/')


def print_columns_with_na(df: pd.DataFrame, print_results: bool=True):
    '''
    Поиск пропусков в данных
    '''
    nulls = pd.DataFrame(df.isnull().sum(), columns=['count'])
    nulls['percent'] = nulls['count'] * 100 / df.shape[0]
    with_nulls = nulls[nulls['count'] > 0]
    if print_results:
        print('Пропусков нет' if len(with_nulls) == 0 else with_nulls)
    return with_nulls


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        raise RuntimeError(f'Пути {DATASET_PATH} не существует')
    
    # Загрузка датасета
    filepath = os.path.join(DATASET_PATH, 'train.csv')
    df_train = pd.read_csv(filepath, delimiter = ',')
    if df_train is None or not len(df_train):
        raise RuntimeError('Датасет не считан')
    
    print('--> Сводная информация по структуре датасета:')
    print(df_train.info())

    # Разбиение на тренировочную и валидационную
    X = df_train.drop('Price(euro)', axis=1)
    y = df_train['Price(euro)']

    # Надо еще разбить для модели
    df_train, df_test = train_test_split(
        df_train,
        test_size=0.3, 
        random_state=48)
    
    X_train = df_train.drop('Price(euro)', axis=1)
    y_train = df_train['Price(euro)']
    
    X_test = df_test.drop('Price(euro)', axis=1)
    y_test = df_test['Price(euro)']

    # Создание и обучение модели
    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=15)
    model.fit(X, y, eval_set=[(X_train, y_train), (X_test, y_test)])

    # Для примера сравним с моделью линейной
    model_linear = SGDRegressor(random_state=48)
    model_linear.fit(X, y)

    dump(model, 
         filename=os.path.join(DATASET_PATH, 'xgbregressor.joblib'))
    print('--> Модель XGBRegressor сохранена')

    dump(model_linear, 
         filename=os.path.join(DATASET_PATH, 'sgdregressor.joblib'))
    print('--> Модель SGDRegressor сохранена')
