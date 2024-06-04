import os
import warnings
import pandas as pd

from joblib import load
from pydantic import BaseModel

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse, PlainTextResponse


MODEL_PATH: str = os.environ.get('MODEL_PATH', './models')

APP = FastAPI()

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


class OutputData(BaseModel):
    '''
    Выходные данные модели
    '''
    salary: float 


def _load_model_and_pipeline(model_path: str, model_name: str='data1') -> tuple[object, object]:
    '''
    Поиск и загрузка модели, и трансформера для модели
    '''
    models = sorted([name for name in os.listdir(model_path) \
                     if name.endswith(".joblib") \
                        and name.startswith(model_name)])
    if not models:
        raise RuntimeError("В указанном каталоге нет моделей")
    
    model = load(f'{os.path.join(model_path, models[0])}')
    if model is None:
        raise RuntimeError(f"Не найдена модель для {models[0]}")
    
    transformer = load(f'{os.path.join(model_path, f"transformer_{models[0]}")}')
    if transformer is None:
        raise RuntimeError(f"Не найден трансформер для {models[0]}")

    return model, transformer


def _model_predict(data: InputData, model, transformer) -> OutputData:
    '''
    Получение целевого значения
    '''
    if model is None:
        raise RuntimeError(f"Модель не может быть пустой")
    if transformer is None:
        raise RuntimeError(f"Трансформер не может быть пустой")

    data_df = pd.DataFrame([data.model_dump()])
    data_df.columns = data_df.columns.str.upper()

    trans_df = transformer.transform(data_df)
    pred_salary = model.predict(trans_df)
    if pred_salary is None:
        pred_salary = .0
    return OutputData(salary=float(pred_salary[0]))



@APP.get('/check')
async def check_handler() -> PlainTextResponse:
    """
    Пустой метод для проверки состояния
    """
    return PlainTextResponse(content="I`m Ok!", status_code=status.HTTP_200_OK)


@APP.get('/example')
async def example_handler() -> PlainTextResponse:
    """
    Возврат пустой модели для запроса
    """
    data = InputData(
        work_experience_years=15, 
        year=2023, 
        job_title_name="Developer", 
        employment_type_name="Contract", 
        work_model_name="On-Site", 
        work_grade_name="Architect")

    return JSONResponse(
        content=data.model_dump(),
        status_code=status.HTTP_200_OK
    )


@APP.post('/predict')
async def predict_handler(data: InputData) -> JSONResponse:
    """
    Предсказание ЗП
    """
    model, transformer = _load_model_and_pipeline(MODEL_PATH)

    return JSONResponse(
        content=_model_predict(data, model, transformer).model_dump(),
        status_code=status.HTTP_200_OK
    )


if __name__ == '__main__':
    print('Hello Fast API')
