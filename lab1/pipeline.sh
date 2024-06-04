#!/bin/bash

# Этап 5: Напишите bash-скрипт (pipeline.sh), 
# последовательно запускающий все python-скрипты.


# Установка зависимостей 
echo "Установка зависемостей"
pip3 install -r ./requirements.txt --no-cache-dir

# Этап 1: Генерация данных
echo "Генерация данных"
python3 ./src/data_creation.py

# Этап 2: Предобработка данных для модели
echo "Предобработка данных для модели"
python3 ./src/model_preprocessing.py

# Этап 3: Обучение модели
echo "Обучение модели"
python3 ./src/model_preparation.py

# Этап 4: Тест модели и оценка метрик
echo "Тест модели и оценка метрик"
python3 ./src/model_testing.py
