import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from phik import phik_matrix
from utils.api_client import *

def show_page():
    st.header("Анализ данных")
    uploaded_file = st.file_uploader("Загрузите датасет", type=["csv"])
    st.session_state.train = None
    st.session_state.df = None

    if uploaded_file is not None:
        '''
        dtypes_of_data = {
            'url_car': str,
            'car_make': str,
            'car_model': str,
            'car_gen': str,
            'car_type': str,
            'car_compl': str,
            'ann_id': str,
            'car_price': float,
            'ann_city': str,
            'link_cpl': str,
            'avail': str,
            'year': int,
            'mileage': int,
            'color': str,
            'eng_size': float,
            'eng_power': float,
            'eng_power_kw': float,
            'eng_type': str,
            'pow_resrv': str,
            'options': str,
            'transmission': str,
            'drive': str,
            'st_wheel': str,
            'condition': str,
            'count_owner': str,
            'original_pts': str,
            'customs': str,
            'url_compl': str,
            'state_mark': str,
            'class_auto': str,
            'door_count': float,  # могут быть пропуски
            'seat_count': str,  # могут быть значения с диапазоном, пропуски
            'long': float,
            'width': float,
            'height': float,
            'clearence': str,  # могут быть значения с диапазоном
            'v_bag': str,  # могут быть значения с диапазоном, пропуски
            'v_tank': float,
            'curb_weight': float,
            'gross_weight': float,
            'front_brakes': str,
            'rear_brakes': str,
            'max_speed': float,
            'acceleration': float,
            'fuel_cons': float,
            'fuel_brand': str,
            'engine_loc1': str,
            'engine_loc2': str,
            'turbocharg': str,
            'max_torq': float,
            'cyl_count': float  # Могут быть пропуски
        }
        st.session_state.df = pd.read_csv(uploaded_file, dtype=dtypes_of_data)'''
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.df = st.session_state.df.rename(columns={'Unnamed: 0': 'Unnamed'})
        st.write("Просмотр данных:")
        st.dataframe(st.session_state.df)

        st.write("Общая информация о полях набора данных:")
        df_types = pd.DataFrame(st.session_state.df.dtypes)
        df_nulls = st.session_state.df.count()

        df_null_count = pd.concat([df_nulls, df_types], axis=1)
        df_null_count = df_null_count.reset_index()

        # Переименуем поля
        col_names = ["features", "non_null_counts", "types"]
        df_null_count.columns = col_names

        st.write(df_null_count)

        st.write("Основные характеристики числовых признаков:")
        st.write(st.session_state.df.describe(include=(float, int)))

        st.write("Основные характеристики нечисловых признаков:")
        st.write(st.session_state.df.describe(include=object))

        st.write("Распределение признака/целевой переменной:")
        column = st.selectbox("Выберите признак/целевую переменную", st.session_state.df.columns)
        st.bar_chart(st.session_state.df[column].value_counts())

    # Сохранение данных в сессионное состояние

    if "selected_features" not in st.session_state:
        st.session_state.selected_features = None

    if "first_feature" not in st.session_state:
        st.session_state.first_feature = None

    if "second_feature" not in st.session_state:
        st.session_state.second_feature = None

    # Отправка и обработка файла в API
    if st.session_state.df is not None:
        st.header("Отправка и обработка файла в API")

        if st.button("Отправить и обработать"):
            if uploaded_file is not None:
                # Отправка файла
                with st.spinner("Отправка файла в API..."):
                    response_upload = upload_file("api/v1/dataset/upload", uploaded_file)
                    if response_upload.status_code == 200:
                        result_upload = response_upload.json()
                        st.success(result_upload['message'])
                    else:
                        st.error("Ошибка отправки файла на сервер.")
                        st.stop()  # Остановить выполнение при ошибке отправки

                # Предобработка файла
                with st.spinner("Предобработка данных, пожалуйста, подождите..."):
                    response_preprocess = preprocess_data("api/v1/dataset/preprocessing")
                    if response_preprocess.status_code == 200:
                        result_preprocess = response_preprocess.json()
                        st.success(result_preprocess['message'])
                        st.session_state.train = pd.DataFrame(result_preprocess['train'])
                    else:
                        st.error("Ошибка обработки файла на сервере.")
            else:
                st.error("Пожалуйста, загрузите файл перед отправкой.")

    if st.session_state.train is not None:
        st.write("Просмотр обработанных данных:")
        train = st.session_state.train
        st.dataframe(train)
        # Уменьшим избыточную разрядность чисел
        fcols = train.select_dtypes('float').columns
        icols = train.select_dtypes('integer').columns
        train[fcols] = train[fcols].apply(pd.to_numeric, downcast='float')
        train[icols] = train[icols].apply(pd.to_numeric, downcast='integer')

        # Составим список числовых и категориальных признаков
        num_features = (
            train.select_dtypes(
                include=['integer', 'floating']
            ).columns.to_list()
        )
        cat_features = (
            train.select_dtypes(
                include=['category']
            ).columns.to_list()
        )
        date_features = 'ann_date'

        def create_histogram(feature):
            if feature in date_features:
                train['year_month'] = train[feature].dt.strftime('%Y-%m')
                fig = go.Figure([
                    go.Histogram(
                        x=train['year_month']
                    )
                ])

                fig.update_layout(
                    xaxis_title_text=f'Год и месяц',
                    yaxis_title_text='Количество объектов',
                    hovermode="x"
                )
            elif feature in num_features:
                fig = go.Figure([
                    go.Histogram(
                        x=train[feature],
                        histnorm='percent'
                    )
                ])
                fig.update_layout(
                    xaxis_title_text=f'Значения {feature}',
                    yaxis_title_text='Количество объектов',
                    hovermode="x"
                )
            else:
                unique_values = train[feature].value_counts().to_dict()
                sorted_unique_values = sorted(unique_values.items(),
                                              key=lambda x: x[1], reverse=True)
                if len(sorted_unique_values) <= 20:
                    categories = [x[0] for x in sorted_unique_values]
                    values = [x[1] for x in sorted_unique_values]
                else:
                    categories = [x[0] for x in sorted_unique_values[:20]] + ['other']
                    values = [x[1] for x in sorted_unique_values[:20]]
                    values.append(sum([x[1] for x in sorted_unique_values[20:]]))
                fig = go.Figure([
                    go.Bar(
                        x=categories,
                        y=values
                    )
                ])
                fig.update_layout(
                    xaxis_title_text=f'Значения {feature}',
                    yaxis_title_text='Количество объектов',
                    hovermode="x"
                )

            return fig

        def create_dependency_plot(first_feature, second_feature):
            if first_feature in num_features and second_feature in num_features:
                fig = go.Figure([
                    go.Scatter(
                        x=train[first_feature],
                        y=train[second_feature],
                        mode='markers'
                    )
                ])
                fig.update_layout(
                    yaxis_title_text=f'Второй признак {second_feature}',
                    xaxis_title_text=f'Первый признак {first_feature}',
                    hovermode="x"
                )
            elif first_feature in cat_features and second_feature in cat_features:
                crosstab = pd.crosstab(train[first_feature], train[second_feature])
                fig = px.imshow(crosstab, text_auto=True)
            elif first_feature in date_features or second_feature in date_features:
                if first_feature in date_features:
                    time_col = first_feature
                    other_col = second_feature
                else:
                    time_col = second_feature
                    other_col = first_feature
                train['year_month'] = train[time_col].dt.strftime('%Y-%m')
                if other_col in num_features:
                    fig = px.box(train, x=train['year_month'], y=train[other_col])
                    fig.update_layout(
                        yaxis_title_text=f'Признак {other_col}',
                        xaxis_title_text=f'Год и месяц'
                    )
                else:
                    crosstab = pd.crosstab(train['year_month'], train[other_col])
                    fig = px.imshow(crosstab, text_auto=True)
            else:
                if first_feature in cat_features:
                    cat_col = first_feature
                    num_col = second_feature
                else:
                    cat_col = second_feature
                    num_col = first_feature
                train['categories'] = train[cat_col]
                if train['categories'].nunique() > 20:
                    train['categories'] = train['categories'].astype('object')
                    counts = train[cat_col].value_counts()
                    top_categories = counts.index[:20]
                    train.loc[~train['categories'].isin(top_categories), 'categories'] = 'other'
                    train['categories'] = train['categories'].astype('category')
                fig = px.box(train, x=train['categories'], y=train[num_col])
                fig.update_layout(
                    yaxis_title_text=f'Признак {num_col}',
                    xaxis_title_text=f'Признак {cat_col}'
                )

            return fig

        def get_select_features(threshold=500):
            if 'categories' in train.columns:
                features = train.drop(['categories'], axis=1).columns.to_list()
            else:
                features = train.columns.to_list()

            columns = {'Признак': [], 'Уникальных значений': [], 'Выбран': []}

            for feature in features:
                if feature in num_features:
                    columns['Признак'].append(feature)
                    columns['Уникальных значений'].append('числовой')
                    columns['Выбран'].append(True)
                elif feature in date_features:
                    columns['Признак'].append(feature)
                    columns['Уникальных значений'].append('временной')
                    columns['Выбран'].append(False)
                else:
                    unique_count = train[feature].nunique()
                    columns['Признак'].append(feature)
                    columns['Уникальных значений'].append(str(unique_count))
                    columns['Выбран'].append(unique_count < threshold)

            selection_df = pd.DataFrame(columns)
            selection_df.set_index('Признак', inplace=True)
            if (selection_df['Выбран'] == False).sum() > 0:
                st.write('Исключены следующие признаки:')
                st.dataframe(selection_df.loc[selection_df['Выбран'] == False, 'Уникальных значений'])

            selected_features = selection_df[selection_df['Выбран'] == True].index.to_list()

            return selected_features

        # Выбор признаков
        message, select1, select2 = st.columns(3, vertical_alignment="bottom")
        message.markdown('Выберите два признака для исследования')

        # Селекторы признаков с сохранением выбора в сессионное состояние
        st.session_state.first_feature = select1.selectbox(
            'Первый признак:',
            list(train.columns),
            index=train.columns.tolist().index(st.session_state.first_feature)
            if st.session_state.first_feature in train.columns else 0
        )

        st.session_state.second_feature = select2.selectbox(
            'Второй признак:',
            list(train.columns),
            index=train.columns.tolist().index(st.session_state.second_feature)
            if st.session_state.second_feature in train.columns else 0
        )

        first_feature = st.session_state.first_feature
        second_feature = st.session_state.second_feature

        # Построение графиков
        st.subheader(f'Распределение значений признака {first_feature}')
        fig_hist_first = create_histogram(first_feature)
        st.plotly_chart(fig_hist_first, use_container_width=True)

        if second_feature != first_feature:
            st.subheader(f'Распределение значений признака {second_feature}')
            fig_hist_second = create_histogram(second_feature)
            st.plotly_chart(fig_hist_second, use_container_width=True)

        st.subheader('Зависимость между двумя признаками')
        fig_dependency = create_dependency_plot(first_feature, second_feature)
        st.plotly_chart(fig_dependency, use_container_width=True)

        st.subheader('Корреляция между признаками')

        # Кнопка расчета корреляции с сохранением результата
        if st.button('Рассчитать корреляцию'):
            st.session_state.selected_features = get_select_features()
            corr_matrix = phik_matrix(train[st.session_state.selected_features],
                                      interval_cols=num_features)
            st.session_state.corr_matrix = corr_matrix

        # Отображение сохраненной корреляции
        if "corr_matrix" in st.session_state:
            st.write(st.session_state.corr_matrix)

            fig_corr_heatmap = px.imshow(st.session_state.corr_matrix,
                                         color_continuous_scale='RdBu_r',
                                         zmin=-1, zmax=1, text_auto='.2f')
            st.plotly_chart(fig_corr_heatmap, use_container_width=True)
