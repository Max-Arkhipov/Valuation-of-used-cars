import os
import time
import pandas as pd
import numpy as np

from fastapi import HTTPException
from fastapi.responses import FileResponse
from sklearn.model_selection import train_test_split

from app.preprocessing import preproc
from app.class_model import FullModel
from app.preprocessing_x import preproc_x
import logging

logger = logging.getLogger("file-logger")

models = {}
is_log_models = []
datasets = {}
datasets_prep = {}
learning_curves = {}
loaded_model = None
preproc_pipeline = None


async def upload_csv_dataset(file):
    logger.info(f"Загрузка CSV: имя файла = {file.filename}")
    try:
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())
        df = pd.read_csv(temp_file_path)
        datasets["current"] = df
        os.remove(temp_file_path)
        logger.info("CSV файл успешно загружен")
        return {
            "message": "CSV файл успешно загружен",
            "df.isnull": df.isnull().sum().to_dict()
        }
    except Exception as e:
        logger.error(f"Не удалось обработать CSV файл: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Не удалось обработать CSV файл: {str(e)}")


def perform_eda():
    logger.info("Запущен EDA на текущем наборе данных")
    if "current" not in datasets:
        logger.error("Отсутствует загруженный набор данных для EDA")
        raise HTTPException(status_code=404, detail="Набор данных не загружен")
    df = datasets["current"]
    result = {"statistics": df.describe().to_dict()}
    logger.info("EDA успешно выполнен")
    return result


def preprocessing_data():
    global preproc_pipeline
    logger.info("Начата предобработка данных")
    if "current" not in datasets:
        logger.error("Отсутствует загруженный набор данных для предобработки")
        raise HTTPException(status_code=404, detail="Набор данных не загружен")
    try:
        df = datasets["current"]
        train, test, preproc_pipeline = preproc(df)
        datasets_prep["current"] = [train, test]
        logger.info("Предобработка данных завершена")
        return {
            "message": "Предобработка данных завершена",
            "train": train.to_dict()
        }
    except Exception as e:
        logger.error(f"Ошибка при предобработке данных: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Предобработка не удалась: {e}")


def train_model(config):
    logger.info(f"Начато обучение модели id={config.id}, тип={config.ml_model_type}")
    if config.id in models:
        logger.error(f"Модель с id='{config.id}' уже существует")
        raise HTTPException(status_code=400, detail=f"Модель '{config.id}' уже существует")
    try:
        if config.ml_model_type == "full":
            model = FullModel(config.hyperparameters)
            X_train, y_train, X_test, y_test = model.prepare_data(
                datasets_prep["current"][0], datasets_prep["current"][1]
            )

            logger.info(f"Построение pipeline и обучение модели id={config.id}")
            model.build_pipeline(X_train)
            start_time = time.time()
            model.train_model(X_train, y_train)
            duration = time.time() - start_time

            r2 = model.evaluate_model(X_test, y_test)
            predictions = model.predict(X_test)

            models[config.id] = model
            is_log_models.append(config.id)
            learning_curves[config.id] = model.learning_curve(
                X_train, y_train, X_test, y_test
            )
            logger.info(f"Модель '{config.id}' обучена успешно, R2={round(r2, 4)}, время обучения={duration:.2f}s")
        else:
            logger.error(f"Тип модели '{config.ml_model_type}' не поддерживается")
            raise HTTPException(status_code=400, detail=f"Тип модели '{config.ml_model_type}' не поддерживается")

        return {
            "message": f"Модель '{config.id}' обучена успешно, R2={round(r2, 4)}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Обучение модели '{config.id}' завершилось ошибкой: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Обучение не удалось: {e}")


def load_model_endpoint(request):
    global loaded_model
    model_id = request.id
    logger.info(f"Запрошена загрузка модели id={model_id}")
    if model_id not in models:
        logger.error(f"Модель '{model_id}' не найдена для загрузки")
        raise HTTPException(status_code=404, detail="Модель не найдена")
    loaded_model = models[model_id]
    logger.info(f"Модель '{model_id}' успешно загружена")
    return {"message": f"Модель '{model_id}' загружена"}


def unload_model_endpoint():
    global loaded_model
    logger.info("Запрошена выгрузка текущей модели")
    if not loaded_model:
        logger.error("Нет загруженной модели для выгрузки")
        raise HTTPException(status_code=400, detail="Нет загруженной модели")
    loaded_model = None
    logger.info("Текущая модель успешно выгружена")
    return {"message": "Модель выгружена"}


def list_learning_curve(model_id):
    logger.info(f"Запрошены данные кривой обучения для модели id={model_id}")
    if model_id not in models:
        logger.error(f"Модель '{model_id}' не найдена для построения кривой")
        raise HTTPException(status_code=404, detail="Модель не найдена")
    return learning_curves[model_id]


def make_prediction(model_id, data):
    logger.info(f"Запрошено предсказание для модели id={model_id}, данные={data}")
    global preproc_pipeline
    if model_id not in models:
        logger.error(f"Модель '{model_id}' не найдена для предсказания")
        raise HTTPException(status_code=404, detail=f"Модель '{model_id}' не найдена")
    if preproc_pipeline is None:
        logger.error("Предсказание не выполнено: пайплайн предобработки не инициализирован")
        raise HTTPException(status_code=500, detail="Пайплайн предобработки не инициализирован")
    try:
        input_data = pd.DataFrame([data])
    except Exception as e:
        logger.error(f"Неверный формат входных данных: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Неверный формат входных данных: {e}")

    try:
        processed_data = preproc_pipeline.transform(input_data)
        raw_preds = models[model_id].predict(processed_data)
        if model_id in is_log_models:
            predictions = np.exp(raw_preds)
        else:
            predictions = raw_preds
        logger.info(f"Предсказание для модели '{model_id}' выполнено: {predictions.tolist()}")
        return {"predictions": predictions.tolist()}

    except Exception as e:
        logger.error(f"Ошибка при предсказании для модели '{model_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {e}")


async def predict_items(file):
    logger.info(f"Запрошено пакетное предсказание, файл = '{file.filename}'")
    global loaded_model
    if not loaded_model:
        logger.error("Пакетное предсказание не выполнено: нет загруженной модели")
        raise HTTPException(status_code=400, detail="Нет загруженной модели")
    try:
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())
        df = pd.read_csv(temp_file_path)
        os.remove(temp_file_path)
        df['predict'] = pd.Series(loaded_model.predict(preproc_x(df)))
        df.to_csv('predictions.csv', index=False)
        logger.info("Пакетное предсказание успешно выполнено, возвращаем CSV")
        return FileResponse(
            path='predictions.csv',
            media_type='text/csv',
            filename='predictions.csv'
        )
    except Exception as e:
        logger.error(f"Пакетное предсказание завершилось ошибкой: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Пакетное предсказание не удалось: {e}")


def list_models():
    logger.info("Запрошен список всех моделей")
    return {"models": list(models.keys())}


def remove_model(model_id: str):
    logger.info(f"Запрошено удаление модели id={model_id}")
    if model_id not in models:
        logger.error(f"Удаление не выполнено: модель '{model_id}' не найдена")
        raise HTTPException(status_code=404, detail="Модель не найдена")
    del models[model_id]
    del learning_curves[model_id]
    is_log_models.remove(model_id)
    logger.info(f"Модель '{model_id}' удалена")
    return {"message": f"Модель '{model_id}' удалена"}


def remove_all_models():
    logger.info("Запрошено удаление всех моделей")
    models.clear()
    learning_curves.clear()
    is_log_models.clear()
    logger.info("Все модели успешно удалены")
    return {"message": "Все модели удалены"}
