# Этап 2. Создайте python-скрипт (model_preprocessing.py), который выполняет 
# предобработку данных, например с помощью sklearn.preprocessing.StandardScaler.


import os

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from joblib import dump


#############################################################################
# Основные функции
#############################################################################
def save_dataset(df: pd.DataFrame, name: str, isTest:bool = False) -> None:
    '''Сохранение набора данных'''
    path = "data/train"
    if isTest:
        path = "data/test"
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(f"{path}/{name}.csv", index=False)


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame
                    ) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    '''Функция предварительной обработки данных'''
    # В учебных целях мудрить не будем, сделаем базовые преобразования
    # - One-Hot кодирование для признаков с малым количеством значений:
    #   + JOB_TITLE_NAME	
    #   + EMPLOYMENT_TYPE_NAME	
    #   + WORK_MODEL_NAME	
    #   + WORK_GRADE_NAME
    # - Стандартизацию для признаков:
    #   + WORK_EXPERIENCE_YEARS
    #   + YEAR
    #   + SALARY


    # Перед предварительнойо бработкой надо слить датасеты,
    # иначе может не корректно просчитаться коэффициент
    # ИМХО: Делитьн на наборы надо именно тут!
    full_df = pd.concat([train_df, test_df]) 
    y_train = train_df['SALARY']
    y_test = test_df['SALARY']
    full_df = full_df.drop('SALARY', axis=1)

    one_hot_cols: list = ['JOB_TITLE_NAME', 'EMPLOYMENT_TYPE_NAME', 'WORK_MODEL_NAME', 'WORK_GRADE_NAME']
    stand_scaler_cols: list = ['WORK_EXPERIENCE_YEARS', 'YEAR']

    # Создание pipeline для One-Hot кодирования
    pipe_one_hot = Pipeline([
        ('encoder', OneHotEncoder(
            drop='if_binary', 
            handle_unknown='ignore', 
            sparse_output=False
        ))
    ])

    # Создание pipeline для стандартизации
    pipe_stand = Pipeline([
        ('scaler', StandardScaler())
    ])


    # Создание трансформера колонок данных (далее в этом контексте "трансформер")
    preprocessors = ColumnTransformer(transformers=[
        ('stand_cols', pipe_stand, stand_scaler_cols),
        ('one_hot_cols', pipe_one_hot, one_hot_cols)
    ])
    # Обучение трансформера
    preprocessors.fit(full_df)

    # Соберем имена колонок данных после трансформаций
    trans_cols_list = []

    # Колонки при кодировании числовых признаков не изменились
    trans_cols_list.extend(stand_scaler_cols)
    
    # Колонки при One-Hot кодировании добавляются с новыми именами
    for trans in preprocessors.transformers_:
        if trans[0] not in ['one_hot_cols']:
            continue
        pipe: Pipeline = trans[1]
        if 'encoder' in pipe.named_steps:
            trans_cols_list.extend(list(pipe.get_feature_names_out()))


    # Трансформация данных
    copy_train_df = pd.DataFrame(preprocessors.transform(train_df.copy(deep=True)),
                                 columns=trans_cols_list)
    copy_train_df['SALARY'] = y_train
    copy_test_df = pd.DataFrame(preprocessors.transform(test_df.copy(deep=True)),
                                columns=trans_cols_list)
    copy_test_df['SALARY'] = y_test

    return copy_train_df, copy_test_df, preprocessors


if __name__ == '__main__':
    # Чтение данных из наборов данных
    train_file_list = sorted([name for name in os.listdir('./data/train/') \
                              if name.endswith(".csv") and '_clear' not in name])

    for data_file in train_file_list:
        print(f'-> Обработка файлов: {data_file}')
        if not os.path.exists('./data/test/' + data_file):
            print('--> Связанный файл с валидационными данными не найден')
            continue
        data_train_df = pd.read_csv(f"./data/train/{data_file}")
        data_test_df = pd.read_csv(f"./data/test/{data_file}")

        data_train_df, data_test_df, preprocessors = preprocess_data(data_train_df, data_test_df)

        save_dataset(data_train_df,  data_file.replace(".csv", "_clear", 1))
        save_dataset(data_test_df,  data_file.replace(".csv", "_clear", 1), isTest=True)
        
        # Сохранение трансформера для потомковка данных для
        if not os.path.exists('./models/'):
            os.makedirs('./models/')
        dump(preprocessors, f'./models/transformer_{data_file.replace(".csv", ".joblib", 1)}')
