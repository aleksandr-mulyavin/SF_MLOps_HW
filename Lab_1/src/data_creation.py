# Этап 1: Создайте python-скрипт (data_creation.py), 
# который создает различные наборы данных, описывающие некий 
# процесс (например, изменение дневной температуры). 
# Таких наборов должно быть несколько, в некоторые данные можно 
# включить аномалии или шумы. Часть наборов данных должна быть 
# сохранена в папке «train», другая часть — в папке «test».


import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


#############################################################################
# Константы
#############################################################################

# Константы общие
RANDOM_STATE = 42
SAMPLE_SIZE = 100_000

# Диапазон лет
YEAR_FROM = 2020
YEAR_TO = 2024

# Константы - Стаж работы
WORK_EXPERIENCE_FROM = 0
WORK_EXPERIENCE_TO = 20

# Грейды стажа
WORK_GRADES_DF = pd.DataFrame.from_dict([
    {"name": "Young Padawan", "from": 0, "to": 1, "koef": 0.6, "year_koef": 0.0},
    {"name": "Junior", "from": 1, "to": 3, "koef": 0.8, "year_koef": 0.0},
    {"name": "Middle", "from": 3, "to": 7, "koef": 1.1, "year_koef": 0.05},
    {"name": "Senior", "from": 7, "to": 12, "koef": 1.5, "year_koef": 0.08},
    {"name": "Architect", "from": 12, "to": 100, "koef": 1.9, "year_koef": 0.15},
]) 
WORK_GRADES_DF.columns=["WORK_GRADE_NAME", "WORK_GRADE_YEAR_FROM", "WORK_GRADE_YEAR_TO", "WORK_GRADE_KOEF", "WORK_GRADE_YEAR_KOEF"]

# Модель работы
WORK_MODELS_DF = pd.DataFrame.from_dict([
    {"name": "On-Site", "koef": 1.0},
    {"name": "Remote",  "koef": 0.7},
    {"name": "Hybrid",  "koef": 0.85}
])
WORK_MODELS_DF.columns=["WORK_MODEL_NAME", "WORK_MODEL_KOEF"]

# Тип занятости
EMPLOYMENT_TYPES_DF = pd.DataFrame.from_dict([
    {"name": "Full-Time", "koef": 1.0},
    {"name": "Contract", "koef": 1.2}
])
EMPLOYMENT_TYPES_DF.columns=["EMPLOYMENT_TYPE_NAME", "EMPLOYMENT_TYPE_KOEF"]

# Должности
JOB_TITLES_DF = pd.DataFrame.from_dict([
    {"name": "Data Engineer", "base": 2000.0},
    {"name": "BI Developer", "base": 1750.0},
    {"name": "Developer", "base": 1640.0},
])
JOB_TITLES_DF.columns=["JOB_TITLE_NAME", "JOB_TITLE_BASE"]

# Параметры шума по целевой переменной
SALARY_NOISE_LOW = -0.3
SALARY_NOISE_HIGH = 0.3

# Параметры шума по параметрам
WORK_EXPERIENCE_NOISE_LOW = -1
WORK_EXPERIENCE_NOISE_HIGH = 1


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


def calc_salary(row) -> float:
    '''Расчет ЗП'''
    salary = row["JOB_TITLE_BASE"]
    salary = salary * row["EMPLOYMENT_TYPE_KOEF"]
    salary = salary * row["WORK_MODEL_KOEF"]    # Да, удаленщики получают чуть ниже офисников
    salary = salary * row["WORK_GRADE_KOEF"]
    salary = salary + (salary * row["WORK_GRADE_YEAR_KOEF"]) * (row["WORK_EXPERIENCE_YEARS"] - row["WORK_GRADE_YEAR_FROM"])
    return salary


def gen_base_dataset() -> pd.DataFrame:
    '''Создание базового набора данных'''
    
    # Настройка рандомизатора
    np.random.seed(RANDOM_STATE)

    # Генерация Диапазона лет
    year_keys = np.random.randint(
        YEAR_FROM, 
        YEAR_TO, 
        size=SAMPLE_SIZE)

    # Генерация Стажа работы
    work_experience_keys = np.random.randint(
        WORK_EXPERIENCE_FROM, 
        WORK_EXPERIENCE_TO, 
        size=SAMPLE_SIZE)

    # Расчет Грейдов
    work_grade_keys = np.zeros(
        shape=(SAMPLE_SIZE),
        dtype=int
    )

    for i in range(SAMPLE_SIZE):
        work_experience: int = work_experience_keys[ i ]
        exp_from_cl = WORK_GRADES_DF["WORK_GRADE_YEAR_FROM"] <= work_experience
        exp_to_cl = WORK_GRADES_DF["WORK_GRADE_YEAR_TO"] >= work_experience
        result_index_list = WORK_GRADES_DF.index[exp_from_cl & exp_to_cl]
        if len(result_index_list) > 0:
            work_grade_keys[ i ] = result_index_list[ 0 ]
        else:
            print(f"{work_experience}")
            work_grade_keys[ i ] = np.NAN

    # Генерация Модели работы
    work_model_keys = np.random.randint(
        0, 
        len(WORK_MODELS_DF), 
        size=SAMPLE_SIZE)
    
    # Генерация Типа занятости
    employment_type_keys = np.random.randint(
        0, 
        len(EMPLOYMENT_TYPES_DF), 
        size=SAMPLE_SIZE)
    
    # Генерация Должности
    job_title_keys = np.random.randint(
        0, 
        len(JOB_TITLES_DF), 
        size=SAMPLE_SIZE)


    # Создание единого датасета
    df = pd.DataFrame(
        data=np.array([job_title_keys, employment_type_keys, work_model_keys, work_grade_keys, work_experience_keys, year_keys]).T,
        columns=['JOB_TITLE_KEYS', 'EMPLOYMENT_TYPE_KEYS', 'WORK_MODEL_KEYS', 'WORK_GRADE_KEYS', 'WORK_EXPERIENCE_YEARS', 'YEAR']
    )

    # Соединение со справочником JOB_TITLES
    full_df = df.merge(JOB_TITLES_DF, left_on="JOB_TITLE_KEYS", right_index=True) \
        .merge(EMPLOYMENT_TYPES_DF, left_on="EMPLOYMENT_TYPE_KEYS", right_index=True) \
        .merge(WORK_MODELS_DF, left_on="WORK_MODEL_KEYS", right_index=True) \
        .merge(WORK_GRADES_DF, left_on="WORK_GRADE_KEYS", right_index=True) \
        .drop(columns=["JOB_TITLE_KEYS", "EMPLOYMENT_TYPE_KEYS", "WORK_MODEL_KEYS", "WORK_GRADE_KEYS"])

    # Рассчитаем и добавим ЗП
    full_df['SALARY'] = full_df.apply(calc_salary, axis=1)

    # Удаление служебных полей 
    # В теории мы можем знать коэффициенты 
    # расчета ЗП, но сейчас они нам не нужны
    full_df.drop(['JOB_TITLE_BASE', 'EMPLOYMENT_TYPE_KOEF',
                'WORK_MODEL_KOEF', 'WORK_GRADE_YEAR_FROM',
                'WORK_GRADE_YEAR_TO', 'WORK_GRADE_KOEF',
                'WORK_GRADE_YEAR_KOEF'],
                axis=1,
                inplace=True)

    return full_df


def get_dataset_1(base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Формирование набора 1'''
    # Разбиение на учебную и тестовую выборки
    data1_train_df, data1_test_df = train_test_split(
        base_df, 
        test_size=0.3, 
        random_state=RANDOM_STATE)
    return data1_train_df, data1_test_df


def get_dataset_2(base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Формирование набора 2'''
    # Копирование исходного датасета
    base_2_df = base_df.copy(deep=True)

    # Генерация шума для целевой переменной в диапазоне от 0.3 до -0.3
    salary_noise_koef = np.random.uniform(
        low=SALARY_NOISE_LOW, 
        high=SALARY_NOISE_HIGH, 
        size=SAMPLE_SIZE)
    
    # Добавление шума в отдельный столбец
    base_2_df["SALARY"] = base_2_df["SALARY"] + base_2_df["SALARY"] * salary_noise_koef

    # Разбиение на учебную и тестовую выборки
    data2_train_df, data2_test_df = train_test_split(
        base_2_df, 
        test_size=0.3, 
        random_state=RANDOM_STATE)
    return data2_train_df, data2_test_df


def get_dataset_3(base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Формирование набора 3'''
    # Копирование исходного датасета
    base_3_df = base_df.copy(deep=True)

    # Генерация шума для стажа работы (кто то приверает на 1 или 2 года)
    work_experience_noise_koef = np.random.choice([0, 1, 2], SAMPLE_SIZE, p=[0.9, 0.08, 0.02])

    # Добавление шума в отдельный столбец
    base_3_df["WORK_EXPERIENCE_YEARS"] = base_3_df["WORK_EXPERIENCE_YEARS"] + work_experience_noise_koef

    # Генерация выбросов по ЗП (выбросы почти в разы)
    outliers_koef = np.random.choice([-2.5, -2, 0, 2, 3], SAMPLE_SIZE, p=[0.01, 0.04, 0.9, 0.04, 0.01])

    # Применение выбросов на данные
    base_3_df["SALARY"] = base_3_df["SALARY"] + base_3_df["SALARY"] * outliers_koef / 10

    # Разбиение на учебную и тестовую выборки
    data3_train_df, data3_test_df = train_test_split(
        base_3_df, 
        test_size=0.3, 
        random_state=RANDOM_STATE)
    return data3_train_df, data3_test_df


if __name__ == '__main__':
    # Генерация базового набора
    base_df  =  gen_base_dataset()
    print('-> Базовый набор сгенерирован')

    # Обработка данных для первого набора
    data1_train_df, data1_test_df = get_dataset_1(base_df)
    save_dataset(data1_train_df, "data1")
    save_dataset(data1_test_df,  "data1", isTest=True)
    print('-> Набор 1 сгенерирован')

    # Обработка данных для второго набора
    data2_train_df, data2_test_df = get_dataset_2(base_df)
    save_dataset(data2_train_df, "data2")
    save_dataset(data2_test_df,  "data2", isTest=True)
    print('-> Набор 2 сгенерирован')

    # Обработка данных для третьего набора
    data3_train_df, data3_test_df = get_dataset_3(base_df)
    save_dataset(data3_train_df, "data3")
    save_dataset(data3_test_df,  "data3", isTest=True)
    print('-> Набор 3 сгенерирован')
