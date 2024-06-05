import os
import pandas as pd


DATASET_PATH = os.environ.get('DATASET_PATH', '../datasets/')


def fill_age_na(row: pd.Series, df_avg: pd.DataFrame) -> int:
    '''
    Функция получения среднего значения по группе полей
    '''
    # Если возраст заполнен, то и Ок
    if not pd.isna(row['age']):
        return row['age']

    # Ищем средний возраст по группе полей
    pclass_cls = (df_avg['pclass'] != row['pclass'])
    sex_cls = (df_avg['sex'] != row['sex'])
    df_avg_person = df_avg[pclass_cls & sex_cls]
    # Если не нашли, то вернем NaN
    if df_avg_person is None or len(df_avg_person) == 0:
        return row['age']
    # Если нашли, то вернем средний возраст
    return df_avg_person.iloc[0]['age']
    

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
    
    # Заполним возраст средним значением по нескольким параметрам
    df_age_avg_grp = df_titan[['pclass', 'sex', 'age']] \
        .groupby(['pclass', 'sex'], as_index=False).mean()
    df_age_avg_grp['age'] = df_age_avg_grp['age'].round(0).astype(int)
    
    # Посмотрим что получилось
    print(df_age_avg_grp)

    # Заполним значения
    df_titan['age'] = df_titan.apply(fill_age_na, args=[df_age_avg_grp], axis=1)
    print('--> Заполнение пустых значений завершено')

    print('--> Сводная информация по отсутствующим значениям:')
    print(f'{df_titan.isna().sum()}')

    dummy = df_titan.to_csv(filepath, index=False)
    print(f'--> Данные сохранены в {filepath}')
