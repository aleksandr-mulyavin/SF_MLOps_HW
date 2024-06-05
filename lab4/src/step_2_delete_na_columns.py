import os
import pandas as pd


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
    print('--> Сводная информация по отсутствующим значениям:')
    print(f'{df_titan.isna().sum()}')
    # Не указано какую модификацию сделать, потому 
    # просто удалим несколько признаков
    df_titan = df_titan.drop(['home.dest', 'cabin'], axis=1)
    print('--> Сводная информация по структуре датасета после изменения:')
    print(df_titan.info())
    dummy = df_titan.to_csv(filepath, index=False)
    print(f'--> Данные сохранены в {filepath}')