# Этап 3: Создайте python-скрипт (model_preparation.py), который создает и обучает 
# модель машинного обучения на построенных данных из папки «train».


import os

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from joblib import dump

#############################################################################
# Основные функции
#############################################################################

def train_model(train_df: pd.DataFrame, filename: str) -> None:
    '''Обучение модели и сохранение на диск'''
    # Создание модели логистической регрессии
    model = LinearRegression()

    # Разделение данных на параметры и целевую переменную
    X = train_df.drop(['SALARY'], axis=1)
    y  = train_df['SALARY']

    # Обучение модели
    model.fit(X, y)

    # Сохранение модели
    if not os.path.exists('./models/'):
        os.makedirs('./models/')
    dump(model, f'./models/{filename.replace(".csv", ".joblib", 1)}')


if __name__ == '__main__':
    # Чтение данных из наборов данных (только обработанные)
    train_file_list = [name for name in os.listdir('./data/train/') \
                    if name.endswith(".csv") and '_clear' in name]

    for data_file in train_file_list:
        print(f'-> Обработка файла: {data_file}')
        data_train_df = pd.read_csv(f"./data/train/{data_file}")

        train_model(data_train_df, data_file.replace("_clear",  "",  1))