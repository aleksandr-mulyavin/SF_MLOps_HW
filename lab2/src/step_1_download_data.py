import os
import pandas as pd


DATASET_PATH = os.environ.get('DATASET_PATH', '../datasets/')
DATASET_URL = 'https://raw.githubusercontent.com/dayekb/mpti_ml/main/data/cars.csv'
ADDIT_DATA_URL = 'https://drive.google.com/u/0/uc?id=1LlKrYIKA24birqRgc5PMnpnSmwtxQ6OQ&export=download'


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    
    # Загрузка датасета
    df_cars = pd.read_csv(DATASET_URL, delimiter = ',')
    if df_cars is None or not len(df_cars):
        raise RuntimeError('Датасет не считан')
    
    # Сохранение датасета
    filepath = os.path.join(DATASET_PATH, 'cars.csv')
    df_cars.to_csv(filepath, index=False)
    print(f"--> Данные сохранены в {filepath}")

    # Загрузка доп данных для датасета
    df_addit = pd.read_csv(ADDIT_DATA_URL, delimiter = ';')
    if df_addit is None or not len(df_addit):
        raise RuntimeError('Доп.данные не считаны')
    
    # Сохранение доп данных датасета
    filepath = os.path.join(DATASET_PATH, 'addit.csv')
    df_addit.to_csv(filepath, index=False)
    print(f"--> Доп.данные сохранены в {filepath}")
