# Страница для инференса
import json
import pandas as pd
import streamlit as st
from utils.api_client import *


def show_page():
    st.header("Инференс")
    st.write("Сделайте предсказание для новых данных.")

    # Инициализация состояния
    if "list_model" not in st.session_state:
        st.session_state.list_model = []
    if "model_id" not in st.session_state:
        st.session_state.model_id = None
    if "train" not in st.session_state:
        st.session_state.train = None

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
            "Выберите модель для предсказания:",
            st.session_state.list_model,
            index=st.session_state.list_model.index(st.session_state.model_id)
            if st.session_state.model_id in st.session_state.list_model
            else 0,
        )

    # Проверка наличия загруженных данных
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        df = df.rename(columns={'Unnamed': 'Unnamed: 0'})
        categorical_columns = df.select_dtypes(include="object").columns
        numerical_columns = df.select_dtypes(include=["float", "int"]).columns

        st.subheader("Заполните значения для предсказания:")
        st.session_state.user_input = {}

        # Функция для фильтрации значений по выбранным ранее
        def filter_options(column):
            """Фильтрует значения текущего столбца на основе предыдущих выборов."""
            filtered_df = df.copy()
            for col, val in st.session_state.user_input.items():
                if val is not None and col in categorical_columns:
                    filtered_df = filtered_df[filtered_df[col] == val]
            return [None] + filtered_df[column].dropna().unique().tolist()

        def fill_fields_by_category(df, required_fields, optional_fields, hidden_fields):
            """
            Заполняет данные, разделенные на обязательные, необязательные и скрытые поля.
            Все данные сохраняются в `st.session_state` и отправляются на бэкэнд.

            :param df: DataFrame с данными.
            :param required_fields: Список обязательных полей.
            :param optional_fields: Список необязательных полей.
            :param hidden_fields: Список скрытых полей.
            """
            # Инициализация состояния для пользовательских данных
            if "user_input" not in st.session_state:
                st.session_state.user_input = {}

            user_input = st.session_state.user_input

            # Функция для фильтрации значений по выбранным ранее
            def filter_options(column):
                """Фильтрует значения текущего столбца на основе предыдущих выборов."""
                filtered_df = df.copy()
                for col, val in user_input.items():
                    if val is not None and col in df.select_dtypes(include="object").columns:
                        filtered_df = filtered_df[filtered_df[col] == val]
                return [None] + filtered_df[column].dropna().unique().tolist()

            # Поля для фильтрации по выбранным ранее
            FILTRED_FIELDS = [
                "car_make", "car_model", "car_gen", "car_type",
                "transmission", "drive", "st_wheel", "state_mark",
                "class_auto", "v_bag", "v_tank", "front_brakes", "rear_brakes"
            ]

            # Обязательные поля
            st.markdown("### Обязательные поля")
            cols = st.columns(4)
            for i, col in enumerate(required_fields):
                with cols[i % 4]:
                    if col in df.select_dtypes(include="object").columns:  # Категориальные поля
                        if col in FILTRED_FIELDS:
                            options = filter_options(col)
                        else:
                            options = [None] + df[col].dropna().unique().tolist()
                        user_input[col] = st.selectbox(f"{col}:", options, key=f"required_{col}")
                    elif col in df.select_dtypes(include=["float", "int"]).columns:  # Числовые поля
                        user_input[col] = st.number_input(f"{col}:", value=None, key=f"required_{col}")
                    else:  # Поля без фильтрации
                        user_input[col] = st.text_input(f"{col}:", value="", key=f"required_{col}")

            # Необязательные поля
            st.markdown("### Необязательные поля")
            cols = st.columns(4)
            for i, col in enumerate(optional_fields):
                with cols[i % 4]:
                    if col in df.select_dtypes(include="object").columns:  # Категориальные поля
                        if col in FILTRED_FIELDS:
                            options = filter_options(col)
                        else:
                            options = [None] + df[col].dropna().unique().tolist()
                        user_input[col] = st.selectbox(f"{col}:", options, key=f"optional_{col}")
                    elif col in df.select_dtypes(include=["float", "int"]).columns:  # Числовые поля
                        user_input[col] = st.number_input(f"{col}:", value=None, key=f"optional_{col}")
                    else:  # Поля без фильтрации
                        user_input[col] = st.text_input(f"{col}:", value="", key=f"optional_{col}")

            # Присваивание значений скрытым полям
            for col in hidden_fields:
                user_input[col] = None

            # Сохранение пользовательского ввода
            st.session_state.user_input = user_input

        REQUIRED_FIELDS = [
            "car_make", "car_model", "car_gen", "eng_type",
            "year", "mileage", "color", "count_owner"
        ]
        OPTIONAL_FIELDS = [
            "car_type", "eng_size", "eng_power", "transmission", "drive",
            "st_wheel", "state_mark", "class_auto", "door_count", "seat_count",
            "long", "widht", "height", "clearence", "v_bag", "v_tank",
            "curb_weight", "gross_weight", "front_brakes", "rear_brakes",
            "max_speed", "acceleration", "fuel_cons", "fuel_brand",
            "engine_loc1", "engine_loc2", "turbocharg", "max_torq", "cyl_count"
        ]
        HIDDEN_FIELDS = [
            "Unnamed: 0", "url_car", "car_compl", "ann_date", "ann_id",
            "car_price", "ann_city", "link_cpl", "avail", "eng_power_kw",
            "pow_resrv", "options", "condition", "original_pts", "customs",
            "url_compl"
        ]

        fill_fields_by_category(df, REQUIRED_FIELDS, OPTIONAL_FIELDS, HIDDEN_FIELDS)

        # Кнопка для выполнения предсказания
        if st.button("Сделать предсказание"):
            # st.write(st.session_state.user_input)
            if not st.session_state.model_id:
                st.warning("Сначала выберите модель.")
            else:
                try:
                    # Отправка данных на сервер для предсказания
                    response = make_prediction(
                        "api/v1/models/predict",
                        model_id=st.session_state.model_id,
                        input_data=st.session_state.user_input,
                    )
                    if response and response.status_code == 200:
                        predictions = response.json().get("predictions", [])
                        st.success("Предсказание выполнено успешно!")
                        revenue = predictions[0]
                        st.write("Результат предсказания:", f'{revenue:,.2f}')
                    else:
                        st.error(f"Ошибка: сервер вернул код {response.status_code}")
                except Exception as e:
                    st.error(f"Ошибка выполнения предсказания: {e}")
    else:
        st.warning("Пожалуйста, загрузите и обработайте данные на странице загрузки.")