import os
import pandas as pd


DATASET_PATH = os.environ.get('DATASET_PATH', '../datasets/')
DATASET_URL = 'https://storage.googleapis.com/kagglesdsdata/competitions/10118/111043/' \
    + 'titanic_train.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&' \
    + 'Expires=1717853560&Signature=VQoZHca8nRfzV5N3rNS%2BHSG3CZ0%2FFMebU3zD0x3%2FLRr5' \
    + '85zbB%2B3aJL7RyrUnbxxdSzoWAXcikxoy57Zsgj3Fd4W8H%2FMl2oc1s4GKA2nYeSMVStqeH6MPbTP' \
    + 'dNDaDVkb1SdVzzeMx3dz7YNnLtIBxo8fvK7Upm%2Fx7L%2FLS0nFw2lFtwROT9gmvh8hffs46xrqCIeG' \
    + '6buNX9HFyubo%2FRhjgUhdS68KiijH2g0xR8RY0qvprFg9MbVExETbZlT4womvfmsknwFdnBaJUstWbC' \
    + 'Ks9CHqkdHFy6V7G9dN84wyRGTlsjlciZ0O5TGyvDzj9UkApw57J6aJujBAj0g0odou5QA%3D%3D&response' \
    + '-content-disposition=attachment%3B+filename%3Dtitanic_train.csv'


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        raise RuntimeError(f'Пути {DATASET_PATH} не существует')
    # Загрузка датасета
    df_titan = pd.read_csv(DATASET_URL, delimiter = ',')
    if df_titan is None or not len(df_titan):
        raise RuntimeError('Датасет не считан')
    # Сохранение датасета
    filepath = os.path.join(DATASET_PATH, 'titan.csv')
    df_titan.to_csv(filepath, index=False)
    print(f"--> Данные сохранены в {filepath}")
