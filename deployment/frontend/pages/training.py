# Страница обучения моделей
import streamlit as st
import requests
from utils.api_client import *

def show_page():
    st.header("Обучение модели")
    st.write("Настройте параметры модели и запустите процесс обучения")

    # Инициализация состояния
    if "list_model" not in st.session_state:
        st.session_state.list_model = []
    if "model_id" not in st.session_state:
        st.session_state.model_id = None

    # Задаем параметр Альфа
    alpha = st.slider("Коэффициент регуляризации (alpha)", 0.01, 50.0, 1.0)

    # Задаем идентификатор модели
    model_id = st.text_input("Идентификатор модели", "model_1")

    # Выбор типа модели
    model_type = st.selectbox(
        "Тип модели",
        options=["full", "poly", "ohe"],
        index=0,
        help="Выберите тип модели для обучения"
    )

    # Обучение модели
    if st.button("Обучить модель"):
        if not model_id:
            st.error("Пожалуйста, укажите идентификатор модели.")
        else:
            with st.spinner("Обучение модели, пожалуйста подождите..."):
                response = train_model(model_id, model_type, alpha, "api/v1/models/fit")
            if response.status_code == 200:
                result = response.json()
                st.success(result['message'])
                st.write(f"Сообщение: {result['message']}")

            else:
                st.error(f"Ошибка: {response.json()['detail']}")

    # Кнопка для обновления списка моделей
    if st.button("Обновить список"):
        try:
            response = list_models("api/v1/models/list_models")
            if response and response.status_code == 200:
                st.session_state.list_model = response.json().get("models", [])
                if st.session_state.list_model:
                    st.success("Список моделей обновлен!")
                else:
                    st.warning("Модели не найдены.")
            else:
                st.error(f"Ошибка: сервер вернул код {response.status_code}")
        except Exception as e:
            st.error(f"Ошибка вызова API: {e}")

    # Выбор модели
    if st.session_state.list_model:
        st.session_state.model_id = st.selectbox(
            "Выберите модель для построения кривой обучения:",
            st.session_state.list_model,
            index=st.session_state.list_model.index(st.session_state.model_id)
            if st.session_state.model_id in st.session_state.list_model
            else 0
        )

    # Размещение кнопок в одной строке
    col1, col2 = st.columns(2)

    with col1:
        # Кривая обучения
        if st.button("Показать кривую обучения"):
            response = learning_curve("api/v1/models/learning_curve", st.session_state.model_id)
            if response.status_code == 200:
                learning_curve_data = response.json()
                st.write(f"Кривая обучения для модели '{st.session_state.model_id}':")
                st.line_chart({
                    "r2_score на трейне": learning_curve_data["train_errors"],
                    "r2_score на тесте": learning_curve_data["test_errors"]
                })
            else:
                st.error(f"Ошибка: {response.json()['detail']}")

    with col2:
        # Удаление модели
        if st.button("Удалить модель"):
            try:
                response = delete_model(st.session_state.model_id)
                if response.status_code == 200:
                    st.success(f"Модель '{st.session_state.model_id}' успешно удалена.")
                    # Обновление списка моделей
                    response_list = list_models("api/v1/models/list_models")
                    if response_list and response_list.status_code == 200:
                        st.session_state.list_model = response_list.json().get("models", [])
                    else:
                        st.warning("Не удалось обновить список моделей после удаления.")
                else:
                    st.error(f"Ошибка удаления модели: {response.json()['detail']}")
            except Exception as e:
                st.error(f"Ошибка вызова API для удаления модели: {e}")

