from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import re
import pandas as pd
import numpy as np

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

# Загружаем веса
with open('models/knn_imputer.pkl', 'rb') as file:
    model_imputer = pickle.load(file)
with open('models/my_best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)
with open('models/ohe.pkl', 'rb') as file:
    model_ohe = pickle.load(file)
with open('models/scaler.pkl', 'rb') as file:
    model_scaler = pickle.load(file)


def cast_to_float(df):
    """
    Функция для преобразования колонок
    'mileage', 'engine', 'max_power'
    """

    for col in df[['mileage', 'engine', 'max_power']]:
        df[col] = df[col].str.split(' ', expand=True)[0]
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def torque_func(df):
    """
    Промежуточная функция для преобразования колонки torque к нормальному виду: NM or KGM @ RPM
    Признак NM и KGM выносим в отдельную колонку
    """

    temp_df = df
    temp_df['torque'] = temp_df['torque'].astype(str)
    # Создаем новую колонку, в которую запишем NM или KGM, для дальнейшего перевода к одной единице измерения
    temp_df['torque_indicator'] = temp_df['torque'].apply(
        lambda x: 'NM' if 'nm' in str(x).lower() else 'KGM' if 'kgm' in str(x).lower() else None)
    temp_df['torque'] = temp_df['torque'].replace('at', '@', regex=True).str.strip()
    temp_df['torque'] = temp_df['torque'].apply(lambda x: str(x) + '@' if isinstance(x, str) and '@' not in x else x)
    # С помощью регулярок приводим колонку к однотипному формату:
    temp_df['torque'] = temp_df['torque'].apply(lambda x: re.sub(r'\(.*\)', '', x)).str.strip()
    temp_df['torque'] = temp_df['torque'].apply(lambda x: re.sub(r'nm|NM|Nm|kgm|KGM|rpm|RPM', '', x)).str.strip()
    temp_df['torque'] = temp_df['torque'].replace('~', '-', regex=True).str.strip()
    temp_df['torque'] = temp_df['torque'].replace(',', '', regex=True).str.strip()
    temp_df['torque'] = temp_df['torque'].replace('/', '@', regex=True).str.strip()
    final_df = temp_df
    # Заменяем значения вида хххх+@-500 на хххх, пунктов выше поменял / на @
    final_df['torque'] = final_df['torque'].str.replace(r'\+@-.+', '', regex=True)

    return final_df


def final_torque(df):
    """
    Функция для разделения столбца 'torque' на 2 колонки
    """

    df[['torque', 'max_torque_rpm']] = df['torque'].str.split('@', n=1, expand=True).replace('', np.nan)
    df['max_torque_rpm'] = df['max_torque_rpm'].replace('@', '', regex=True)
    df['torque'] = df['torque'].str.strip()
    df['max_torque_rpm'] = df['max_torque_rpm'].str.strip()
    # Оставляем максимальное значение хххх, где вид уууу-хххх
    df['max_torque_rpm'] = df['max_torque_rpm'].str.split('-').str[-1]
    df['torque'] = df['torque'].astype(float)
    df['max_torque_rpm'] = df['max_torque_rpm'].astype(float)
    final_df = df

    return final_df


def kgm_to_nm_convert(df):
    """
    Функция для конвертации KGM в NM
    """

    if df['torque_indicator'] == 'KGM':
        return df['torque'] * 9.80665
    else:
        return df['torque']


def drop_torque_indicator(df):
    df = df.drop('torque_indicator', axis=1)
    return df


def features_and_target(df):
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']
    return X, y


def ohe_transform(df, ohe, cat_cols):
    """
    Функция для One-Hot кодирования категориальных признаков
    """

    encoded_data = ohe.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out())
    df = df.drop(cat_cols, axis=1)
    final_df = pd.concat([df, encoded_df], axis=1)
    return final_df


def data_preparation(df, model_imputer, model_ohe, model_scaler):
    """
    Функция для преобразования входного датасета
    """
    # Обработка категориальных столбцов
    df_float = cast_to_float(df)
    df_torq = torque_func(df_float)
    df_fin_torq = final_torque(df_torq)
    df_fin_torq['torque'] = df_fin_torq.apply(kgm_to_nm_convert, axis=1)
    df_final = drop_torque_indicator(df_fin_torq)

    # Заполнение пропусков с помощью KNN imputer
    nan_cols = ['mileage', 'engine', 'max_power', 'seats', 'torque', 'max_torque_rpm']
    model_imputer.weights = 'uniform'
    imputed_data = model_imputer.transform(df_final[nan_cols])
    df_final[nan_cols] = pd.DataFrame(imputed_data, index=df_final.index, columns=nan_cols)

    # Добавление новых признаков
    X, y = features_and_target(df_final)
    X['name'] = X['name'].str.split(' ', expand=True)[0].astype('category')
    X['seats'] = X['seats'].astype('category')
    X['year_squared'] = X['year'].apply(lambda x: x ** 2)
    X['is_first_owner'] = X['owner'].apply(lambda x: 1 if x == 'First Owner' else 0)

    # Кодирование категориальных признаков
    categ_cols = [col for col in X.columns if X[col].dtype in ['category', 'object']]
    X = ohe_transform(X, model_ohe, categ_cols)

    # Масштабирование
    scaled_cols = X.columns[:8].tolist()
    X[scaled_cols] = pd.DataFrame(model_scaler.transform(X[scaled_cols]), columns=scaled_cols, index=X.index)

    return X, y


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame.from_dict(pd.json_normalize(item.dict()))
    X, y = data_preparation(df, model_imputer, model_ohe, model_scaler)
    prediction = best_model.predict(X)
    non_log_prediction = np.exp(prediction)
    return non_log_prediction


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    for i, item in enumerate(items):
        items[i] = item.dict()
    df = pd.DataFrame.from_dict(pd.json_normalize(items))
    X, y = data_preparation(df, model_imputer, model_ohe, model_scaler)
    X['predicts'] = best_model.predict(X)
    X['predicts'] = np.exp(X['predicts'])
    return X['predicts'].to_list()
