import os
import time
import warnings
import requests
import pandas as pd
import json

from requests import Response, status_codes
from pydantic import BaseModel


DATA_PATH: str = os.environ.get('DATA_PATH', './data/test')
API_HOST: str = os.environ.get('API_HOST', '127.0.0.1')
API_PORT: str = os.environ.get('API_PORT', '8080')


warnings.filterwarnings('ignore')


class InputData(BaseModel):
    '''
    Входные данные модели
    '''
    work_experience_years: int
    year: int
    job_title_name: str
    employment_type_name: str
    work_model_name: str
    work_grade_name: str


def _find_csv_files(path: str) -> list[str]:
    # Чтение данных из наборов данных (только тестовые обработанные)
    test_file_list = sorted([name for name in os.listdir(path) \
                             if name.endswith(".csv") and '_clear' not in name])
    return test_file_list


if __name__ == '__main__':
    print('Hello Client API')

    test_file_list = _find_csv_files(DATA_PATH)
    if not test_file_list:
        raise RuntimeError("Файлы для примера не найдены")
    
    test_file = test_file_list[0]
    test_df = pd.read_csv(os.path.join(DATA_PATH, test_file))
    if test_df is None or not len(test_df):
        raise RuntimeError(f"Файл {test_file} не прочитан")
    
    url = f'http://{API_HOST}:{API_PORT}/'
    for indx, row in test_df.iterrows():
        time.sleep(3)

        data = InputData(
            work_experience_years=row[0],
            year=row[1],
            job_title_name=row[2],
            employment_type_name=row[3],
            work_model_name=row[4],
            work_grade_name=row[5])
        
        try:
            response: Response = requests.post(f'{url}/predict', json=data.model_dump())
            if response.status_code != 200:
                print(f'Ошибка выполнения запроса по адресу: {url}')
                print(response.json())
                continue
            print(f'--> Status {response.status_code}: ' \
                  + f'responsed salary = {json.loads(response.text)["salary"]}, ' \
                  + f'requested salary = {row[6]:.2f}')
        except Exception as ex:
            print(f'Ошибка выполнения запроса по адресу: {url}')
            print(ex)
