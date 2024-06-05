import os
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


DATASET_PATH = os.environ.get('DATASET_PATH', '../datasets/')


if __name__ == "__main__":
    
    if not os.path.exists(DATASET_PATH):
        raise RuntimeError(f'Пути {DATASET_PATH} не существует')
    
    # Загрузка датасета
    filepath = os.path.join(DATASET_PATH, 'titan.csv')
    df_titan = pd.read_csv(filepath, delimiter = ',')
    if df_titan is None or not len(df_titan):
        raise RuntimeError('Датасет не считан')
    
    print('--> Сводная информация по структуре датасета:')
    print(df_titan.info())

    # Создание pipeline для One-Hot кодирования
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    one_hot_encoder.fit(df_titan[['sex']])
    print(one_hot_encoder.get_feature_names_out())

    # Кодирование признака SeX
    df_sex = pd.DataFrame(
        one_hot_encoder.transform(df_titan[['sex']]).toarray(),
        columns=one_hot_encoder.get_feature_names_out()) 
    
    # Объеденим датасеты
    df_titan = df_titan.drop('sex', axis=1).join(df_sex)

    print('--> Информация о созданном датасете с кодированием пола:')
    print(df_titan.info())

    dummy = df_titan.to_csv(filepath, index=False)
    print(f'--> Данные сохранены в {filepath}')