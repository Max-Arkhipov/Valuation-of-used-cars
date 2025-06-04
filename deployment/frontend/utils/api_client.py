# Клиент для взаимодействия с FastAPI
import os
import streamlit as st
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000")

@st.cache_data
def upload_file(endpoint, file=None):
    url = f"{API_BASE_URL}/{endpoint}"
    files = {'file': (file.name, file.getvalue(), 'text/csv')}
    try:
        response = requests.post(url, files=files)
        return response
    except Exception as e:
        st.error(f"Ошибка при отправке файла через API: {e}")
        return f"Ошибка при отправке файла через API: {e}"

def preprocess_data(endpoint):
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        response = requests.post(url)
        return response
    except Exception as e:
        st.error(f"Ошибка предобработки датасета через API: {e}")
        return f"Ошибка предобработки датасета через API: {e}"

def train_model(model_id, model_type, alpha, endpoint):
    url = f"{API_BASE_URL}/{endpoint}"
    config = {
        "id": model_id,
        "ml_model_type": model_type,
        "hyperparameters": alpha,
    }
    try:
        response = requests.post(url, json={"config": config})
        return response
    except Exception as e:
        st.error(f"Ошибка обучения модели через API: {e}")
        return f"Ошибка обучения модели через API: {e}"

def list_models(endpoint):
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        response = requests.get(url)
        return response
    except Exception as e:
        st.error(f"Ошибка при загрузке списка обученных моделей: {e}")
        return f"Ошибка при загрузке списка обученных моделей: {e}"

def learning_curve(endpoint, model_id):
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        response = requests.post(url, json={"id": model_id})
        return response
    except Exception as e:
        st.error(f"Ошибка при загрузке кривой обучения: {e}")
        return f"Ошибка при загрузке кривой обучения: {e}"

def make_prediction(endpoint, model_id, input_data):
    """
    Отправка данных на сервер для инференса.

    :param endpoint: API-эндпоинт для предсказания.
    :param model_id: ID выбранной модели.
    :param input_data: Данные для инференса в формате JSON.
    :return: Ответ сервера.
    """
    url = f"{API_BASE_URL}/{endpoint}"
    payload = {
        "id": model_id,
        "data": input_data
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Проверка на наличие HTTP-ошибок
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при выполнении предсказания: {e}")
        return None

def upload_dataset(data):
    url = f"{API_BASE_URL}/api/v1/dataset/upload"
    files = {"file": (data.name, data, "text/csv")}
    response = requests.post(url, files=files)
    return response.json()

# Функция для вызова API удаления модели
def delete_model(model_id):
    url = f"{API_BASE_URL}/api/v1/models/remove"
    try:
        response = requests.delete(f"{url}?model_id={model_id}")
        return response
    except Exception as e:
        st.error(f"Ошибка вызова API для удаления модели: {e}")
        return None
