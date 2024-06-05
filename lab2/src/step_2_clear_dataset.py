import os
import pandas as pd
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler     
from sklearn.preprocessing import StandardScaler   
from sklearn.preprocessing import PowerTransformer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split

from joblib import dump


warnings.filterwarnings('ignore')


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


def drop_trash(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Удаление всякого мусора 
    '''
    # Удалим все строки хотя бы с одним значением NaN
    df = df.dropna().reset_index(drop=True)

    # Удалим все дубликаты
    df = df.drop_duplicates().reset_index(drop=True)

    # По указанным графикам видно, что есть выбросы по всем характеристикам
    # Удалим строки исходя из представления о вторичном
    # рынке и ожидаемых характеристиках авто
    MIN_YEAR: int = 1980
    MAX_YEAR: int = 2021
    MIN_DISTANCE: float = 2000.0
    MAX_DISTANCE: float = 0.4e6
    MIN_CAPACITY: float = 100
    MAX_CAPACITY: float = 3000
    MIN_PRICE: float = 200.0
    MAX_PRICE: float = 25_000.0

    # Удалим все записи, у которых год меньше MIN_YEAR
    df_min_year_cond = df[df["Year"] < MIN_YEAR]
    print(f'Будет удалено {len(df_min_year_cond)} записей с годом менее {MIN_YEAR}')
    df = df.drop(df_min_year_cond.index).reset_index(drop=True)

    # Удалим все записи, у которых год больше MAX_YEAR
    df_max_year_cond = df[df["Year"] > MAX_YEAR]
    print(f'Будет удалено {len(df_max_year_cond)} записей с годом более {MAX_YEAR}')
    df = df.drop(df_max_year_cond.index).reset_index(drop=True)

    # Удалим все записи, у которых расстояние менее MIN_DISTANCE
    df_min_distance_cond = df[df["Distance"] < MIN_DISTANCE]
    print(f'Будет удалено {len(df_min_distance_cond)} записей с пробегом менее {MIN_DISTANCE}')
    df = df.drop(df_min_distance_cond.index).reset_index(drop=True)

    # Удалим все записи, у которых расстояние более MAX_DISTANCE
    df_max_distance_cond = df[df["Distance"] > MAX_DISTANCE]
    print(f'Будет удалено {len(df_max_distance_cond)} записей с пробегом более {MAX_DISTANCE}')
    df = df.drop(df_max_distance_cond.index).reset_index(drop=True)

    # Удалим все записи, у которых объем менее MIN_CAPACITY
    df_min_capacity_cond = df[df["Engine_capacity(cm3)"] < MIN_CAPACITY]
    print(f'Будет удалено {len(df_min_capacity_cond)} записей с объемом менее {MIN_CAPACITY}')
    df = df.drop(df_min_capacity_cond.index).reset_index(drop=True)

    # Удалим все записи, у которых объем более MAX_CAPACITY
    df_max_capacity_cond = df[df["Engine_capacity(cm3)"] > MAX_CAPACITY]
    print(f'Будет удалено {len(df_max_capacity_cond)} записей с объемом более {MAX_CAPACITY}')
    df = df.drop(df_max_capacity_cond.index).reset_index(drop=True)

    # Удалим все записи, у которых цена менее MIN_PRICE евро
    df_min_price_cond = df[df["Price(euro)"] < MIN_PRICE]
    print(f'Будет удалено {len(df_min_price_cond)} записей с ценой менее {MIN_PRICE} евро')
    df = df.drop(df_min_price_cond.index).reset_index(drop=True)

    # Удалим все записи, у которых цена более MAX_PRICE евро
    df_max_price_cond = df[df["Price(euro)"] > MAX_PRICE]
    print(f'Будет удалено {len(df_max_price_cond)} записей с ценой более {MAX_PRICE} евро')
    df = df.drop(df_max_price_cond.index).reset_index(drop=True)

    # Удалим объемы двигателя менее литра если это не гибрид
    df_not_hybrid_small_capacity = df[(df['Fuel_type'] != 'Hybrid') & (df['Engine_capacity(cm3)'] < 1000)]
    df = df.drop(df_not_hybrid_small_capacity.index).reset_index(drop=True)

    return df


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        raise RuntimeError(f'Пути {DATASET_PATH} не существует')
    
    # Загрузка датасета
    filepath = os.path.join(DATASET_PATH, 'cars.csv')
    df_cars = pd.read_csv(filepath, delimiter = ',')
    if df_cars is None or not len(df_cars):
        raise RuntimeError('Датасет не считан')
    print('--> Сводная информация по структуре датасета:')
    print(df_cars.info())
    
    # Загрузка датасета
    filepath = os.path.join(DATASET_PATH, 'addit.csv')
    df_addit = pd.read_csv(filepath, delimiter = ',')
    if df_addit is None or not len(df_addit):
        raise RuntimeError('Датасет не считан')
    print('--> Сводная информация по доп. структуре датасета:')
    print(df_addit.info())
    
    # Сделаем объединение исходных данных и данных о стране
    # и регионе производства автомобиля
    df = df_cars.merge(
        df_addit,
        left_on='Make',
        right_on='Brand',
        how='inner')
    # Удалим новый лишний столбец
    df.drop(columns='Brand', inplace=True)
    # Выведем новую структуру
    df.info()

    print('--> Удаление всякого мусора')
    print_columns_with_na(df)
    df = drop_trash(df)
    print_columns_with_na(df)

    # Цену из набора исключим, так как она будет целевым признаком
    # Методы кодирования признаков:
    stand_scaler_cols: list = ['Distance']
    norm_scaler_cols: list = ['Engine_capacity(cm3)']
    step_scaler_cols: list = ['Year']   # Цену убрали
    one_hot_encoding_cols: list = ['Make', 'Style', 'Country',
                                'Region', 'Fuel_type', 'Transmission',
                                'Premium']

    # Создание pipeline для стандартизации
    pipe_stand = Pipeline([
        ('scaler', StandardScaler())
    ])
    # Создание pipeline для нормализации
    pipe_norm = Pipeline([
        ('scaler', MinMaxScaler())
    ])
    # Создание pipeline для степенного преобразования
    pipe_step = Pipeline([
        ('power', PowerTransformer())
    ])
    # Создание pipeline для One-Hot кодирования
    pipe_one_hot = Pipeline([
        ('encoder', OneHotEncoder(
            drop='if_binary',
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])
    # Создание трансформера колонок данных (далее в этом контексте "трансформер")
    preprocessors = ColumnTransformer(transformers=[
        ('stand_cols', pipe_stand, stand_scaler_cols),
        ('norm_cols', pipe_norm, norm_scaler_cols),
        ('step_cols', pipe_step, step_scaler_cols),
        ('one_hot_cols', pipe_one_hot, one_hot_encoding_cols)
    ])

    # Обучение трансформера и преобразование данных
    X_prep = preprocessors.fit_transform(df.drop('Price(euro)', axis=1))
    y = df['Price(euro)']

    # Соберем имена колонок данных после трансформаций
    # Колонки при кодировании числовых признаков не изменились
    trans_cols_list = []
    trans_cols_list.extend(stand_scaler_cols)
    trans_cols_list.extend(norm_scaler_cols)
    trans_cols_list.extend(step_scaler_cols)

    # Колонки при One-Hot кодировании добавляются с новыми именами
    for trans in preprocessors.transformers_:
        if trans[0] not in ['one_hot_cols']:
            continue
        pipe: Pipeline = trans[1]
        if 'encoder' in pipe.named_steps:
            trans_cols_list.extend(list(pipe.get_feature_names_out()))

    # Разбиение на учебную и тестовую выборки
    df_prep = pd.DataFrame(X_prep, columns=trans_cols_list)
    df_prep['Price(euro)'] = y

    train_df, test_df = train_test_split(
        df_prep, 
        test_size=0.3, 
        random_state=48)
    
    # Сохранение наборов данных
    dummy = train_df.to_csv(os.path.join(DATASET_PATH, 'train.csv'), 
                            index=False)
    print(f'--> Данные сохранены в {os.path.join(DATASET_PATH, "train.csv")}')

    dummy = test_df.to_csv(os.path.join(DATASET_PATH, 'test.csv'), 
                            index=False)
    print(f'--> Данные сохранены в {os.path.join(DATASET_PATH, "test.csv")}')

    # Сохранение трансформера
    dump(preprocessors, 
         filename=os.path.join(DATASET_PATH, 'preprocessors.joblib'))
    print('--> Трансформер сохранен')