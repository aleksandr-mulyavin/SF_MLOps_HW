# Этап 4: Создайте python-скрипт (model_testing.py), проверяющий 
# модель машинного обучения на построенных данных из папки «test».


import os

import pandas as pd
import numpy as np

from joblib import load
from sklearn.metrics import r2_score

#############################################################################
# Основные функции
#############################################################################

def model_test(test_df: pd.DataFrame, filename: str) -> float:
    '''Тест модели'''
    # Поиск и загрузка подходящей модели
    model = load(f'./models/{filename.replace(".csv", ".joblib", 1)}')
    if model is None:
        # Это проблема, серьезная проблема
        raise RuntimeError(f"Не найдена модель для {filename}")
    
    # Разделение данных на параметры и целевую переменную
    X = test_df.drop(['SALARY'], axis=1)
    y  = test_df['SALARY']

    # Обучение модели
    y_pred = model.predict(X)

    # Определение метрики
    return r2_score(y, y_pred)


if __name__ == '__main__':
    # Чтение данных из наборов данных (только тестовые обработанные)
    test_file_list = sorted([name for name in os.listdir('./data/test/') \
                             if name.endswith(".csv") and '_clear' in name])

    for data_file in test_file_list:
        print(f'-> Обработка файла: {data_file}')
        data_test_df = pd.read_csv(f"./data/test/{data_file}")

        r2 = model_test(data_test_df, data_file.replace("_clear",  "",  1))
        print(f'--> Оценка модели файла ({data_file.replace("_clear",  "",  1)}):  {r2:.3f}')   
