# Домашние задания по MLOps


## Практическое задание №1 (vo_HW)

Решение Практического задания №1 в каталоге [```/lab1/```](/lab1/):
* Скрипт запуска пайплайна [```/lab1/pipeline.sh```](/lab1/pipeline.sh)
* Список зависимостей для окружения [```/lab1/requirements.txt```](/lab1/requirements.txt)
* Файлы Py в каталоге [```/lab1/src/```](/lab1/src/)

Запускать нужно из каталога [```/lab1/```](/lab1/)


### Описание данных и этапы работы

1. Скрипт ```data_creation.py``` генерирует данные о зарплатах программистов. Учитываются такие параметры как:
    * Год сбора данных
    * Опыт работы сотрудника
    * Грейд сотрудника (подаван, джун, мидл, сеньер, архитектор)
    * Модель работы сотрудника (на рабочем месте, удаленно, гибрид)
    * Тип занятости сотрудника (полный день, контрактная работа)
    * Должности (Data Engineer, BI Developer, Developer)
    * Параметры разброса шума (при генерации шума)
    
    Скрипт генерирует позиции справочников и потом их сопоставляет. Алгоритм довольно протой. Расчет суммы ЗП производится по предполагаемой мной схеме и не может соответствовать реальной зависимости ЗП от перечисленных выше параметров.

    Описание наборов данных:
    * В первом наборе нет никаких шумов, зависимость строго линейная.
    * Во втором наборе сгенерирован шум по целевой перменной
    * В третьем наборе сгенерирован шум для стажа работы и шум с выбросами для целевой перменной

2. Скрипт ```model_preprocessing.py``` выполняет стандартизацию числовых параметров и One-Hot кодирование для категориальных признаков. Признаков не много, One-Hot тут не страшен.

3. Скрипт ```model_preparation.py``` обучает модель линейной регрессии на тренировачных данных и сохраняет обученную модель. Тут вероятно имеет смысл почистить от выбросов, но такой задачи не стояло. 

4. Скрипт ```model_testing.py``` производит валидацию модели и выводит метрики в консоль. 


Просмотреть результат выполнения скрипта можно в GitHub Actions:
--> Оценка модели файла (data1.csv):  0.951
--> Оценка модели файла (data2.csv):  0.844
--> Оценка модели файла (data3.csv):  0.818


### Вывод 

Создан скрипт позволяющий генерировать данные и пайплайн проводящий обработку данных с указанной структурой.

Во второй вариации данных шум достаточно сильно снизил метрики линейной модели.

В третьей вариации данных шум и выбросы еще сильнее снизили метрику. В ходе экспериментов установлено, что шумы гораздо сильнее снижали метрику. 


## Практическое задание №3 (vo_HW)

Решение Практического задания №3 в каталоге [```/lab3/```](/lab3/):
* Скрипт развертывания контейнеров [```/lab3/docker-compose.yaml```](/lab3/docker-compose.yaml)
* Модуль генерации данных и модели взят из Практического задания №1 [```/lab1/```](/lab1/)
* Код и скрипт сборки контейнера для API в [```/lab3/api```](/lab3/api/)
* Код и скрипт сборки контейнера для тестового клиента в [```/lab3/client```](/lab3/client/)

### Описание выполненных работ

1. Код подготовки модели и данные для тестового клиенты были взяты из Практического задания №1 [```/lab1/```](/lab1/). Для встраивания в единый скрипт развертывания контейнеров был создан [```Dockerfile```](/lab1/Dockerfile) и настроена сборка с именем сервиса ```mlops-lab3-model```. Данные и модели сохраняются в подключаемое хранилище ```api_model``` и используются в других сервисах. Так как генерация данных занимает время, то в скрипт развертывания контейнеров была встроена проверка на созданный файл и проверка состояния сервиса ```mlops-lab3-model``` для зависимых сервисов.

2. Создан API с помощью библиотеки FastAPI, создан [```Dockerfile```](/lab3/api/Dockerfile) для сборки контейнера, и настроена сборка с именем сервиса ```mlops-lab3-api```. Программа считывает сохраненную модель из подключаемого хранилища ```api_model```. Сервис обрабатывает 3 метода/точки:
    * GET ```/check``` - проверка состояния и получение простого ответа
    * GET ```/example``` - получение примера запроса
    * POST ```/predict``` - получение предсказания по ЗП

    При пазработке данного сервиса не ставилась цель разработал сложную систему валидации запросов и ответов, сервис может не обработать не знакомые ему структуры и данные. Для демонстрации работы сервиса был создан следующий модуль.

3. Создан просто клиент для API, создан [```Dockerfile```](/lab3/client/Dockerfile) для сборки контейнера, и настроена сборка с именем сервиса ```mlops-lab3-client```. Программа считывает сохраненную тестовый набор данных из подключаемого хранилища ```api_model``` и передает построчно для предсказания ЗП.


## Практическое задание №4 (vo_HW)

Решение Практического задания №5 в каталоге [```/lab4/```](/lab4/):
* Скрипты обработки данных находятся в [```/lab4/src/```](/lab4/src/) (запускать из ```/lab4/src/```):
    * [```step_1_download_data.py```](/lab4/src/step_1_download_data.py) - скачивание датасета Титаник
    * [```step_2_delete_na_columns.py```](/lab4/src/step_2_delete_na_columns.py) - удаление колонок 
    * [```step_3_fill_na_age.py```](/lab4/src/step_3_fill_na_age.py) - заполнение пропущенных значений возраста
    * [```step_4_one_hot_sex.py```](/lab4/src/step_4_one_hot_sex.py) - One Hot кодирование поля Пол
* Файл настроек DVC - [```/lab4/.dvc/config```](/lab4/.dvc/config)
* Файл отслеживания версий - [```/lab4/datasets.dvc```](/lab4/datasets.dvc)

Список коммитов фиксации DVC:
* ```c7c8ab4``` Инициализация и настройка DVC, создание скриптов для обработки данных
* ```d43a86c``` DVC Данные скачаны и сохранены
* ```0e96ebc``` DVC Данные модифицированы, удалены параметры
* ```4311eb8``` DVC Данные изменены, заполнены пустые значения возраста
* ```19b4c8d``` DVC Данные изменены, выполнено OneHot кодирование для параметра пола

Дополнительно опробовано переключение между версиями датасета.



## Практическое задание №5 (vo_HW)

Решение Практического задания №5 в каталоге [```/lab5/```](/lab5/):
* Блокнот с заданием [```/lab5/ipynb_tests.ipynb```](/lab5/ipynb_tests.ipynb)

Формируются 3 датасета на подобии тех, что реализованы в задании 1:
* В первом наборе нет никаких шумов, зависимость строго линейная.
* Во втором наборе сгенерирован шум по целевой перменной
* В третьем наборе сгенерирован шум для стажа работы и шум с выбросами для целевой перменной

Модель линейной регрессии обучена на первом наборе (чистом). Далее проведены испытания и получены метрики по всем трем наборам:
* Чистые данные - R2 = 0.92
* Данные с шумами - R2 = 0.80
* С шумами и выбросами - R2 = 0.75

Метрики R2 относительно не высоки, потому что при генерации использованы не равномерно распределенные случайные числа (exponential). Для таких данных нужна модель посерьезнее.