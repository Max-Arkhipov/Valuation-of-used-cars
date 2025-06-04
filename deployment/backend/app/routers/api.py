from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse

from app.services import (
    upload_csv_dataset,
    perform_eda,
    preprocessing_data,
    train_model,
    load_model_endpoint,
    unload_model_endpoint,
    list_learning_curve,
    make_prediction,
    predict_items,
    list_models,
    remove_model,
    remove_all_models,
)
from app.models import (
    DatasetUploadRequest,
    FitRequest,
    LoadRequest,
    LoadResponse,
    UnloadResponse,
    PredictionRequest,
    LearningCurveRequest,
    ModelListResponse,
    RemoveResponse,
)

import logging

router = APIRouter()
logger = logging.getLogger("file-logger")


@router.post("/dataset/upload")
async def upload_csv_dataset_endpoint(file: UploadFile = File(...)):
    logger.info(f"Вызван эндпоинт /dataset/upload, имя файла = {file.filename}")
    if not file.filename.endswith(".csv"):
        logger.error("Загрузка не удалась: файл не является CSV")
        raise HTTPException(status_code=400, detail="Поддерживаются только CSV-файлы")
    result = await upload_csv_dataset(file)
    logger.info("Загрузка CSV успешно выполнена")
    return result


@router.get("/dataset/eda")
def perform_eda_endpoint():
    logger.info("Вызван эндпоинт /dataset/eda")
    result = perform_eda()
    logger.info("EDA выполнен")
    return result


@router.post("/dataset/preprocessing")
def preprocessing_dataset_endpoint():
    logger.info("Вызван эндпоинт /dataset/preprocessing")
    result = preprocessing_data()
    logger.info("Предобработка данных завершена")
    return result


@router.post("/models/fit")
def train_model_endpoint(request: FitRequest):
    logger.info(f"Вызван эндпоинт /models/fit, конфиг: {request.config}")
    result = train_model(request.config)
    logger.info(f"Обучение модели с id = {request.config.id} завершено")
    return result


@router.get("/models/list_models", response_model=ModelListResponse)
def list_models_endpoint():
    logger.info("Вызван эндпоинт /models/list_models")
    result = list_models()
    logger.info(f"Список моделей: {result['models']}")
    return result


@router.post("/models/load", response_model=LoadResponse)
def load_model(request: LoadRequest):
    logger.info(f"Вызван эндпоинт /models/load, id = {request.id}")
    result = load_model_endpoint(request)
    logger.info(f"Модель с id = {request.id} загружена")
    return result


@router.post("/models/unload", response_model=UnloadResponse)
def unload_model():
    logger.info("Вызван эндпоинт /models/unload")
    result = unload_model_endpoint()
    logger.info("Текущая модель выгружена")
    return result


@router.post("/models/predict_items")
async def make_prediction_items_endpoint(file: UploadFile):
    logger.info(f"Вызван эндпоинт /models/predict_items, файл = {file.filename}")
    result = await predict_items(file)
    logger.info("Пакетное предсказание завершено")
    return result


@router.post("/models/learning_curve")
def learning_curves_endpoint(request: LearningCurveRequest):
    logger.info(f"Вызван эндпоинт /models/learning_curve для id = {request.id}")
    result = list_learning_curve(request.id)
    logger.info(f"Отправлена кривая обучения для модели {request.id}")
    return result


@router.post("/models/predict")
def make_prediction_endpoint(request: PredictionRequest):
    logger.info(f"Вызван эндпоинт /models/predict для id = {request.id}, данные = {request.data}")
    result = make_prediction(request.id, request.data)
    logger.info(f"Результаты предсказания для модели {request.id} отправлены")
    return result


@router.delete("/models/remove", response_model=RemoveResponse)
def remove_model_endpoint(model_id: str):
    logger.info(f"Вызван эндпоинт /models/remove для id = {model_id}")
    result = remove_model(model_id)
    logger.info(f"Модель с id = {model_id} удалена")
    return result


@router.delete("/models/remove_all", response_model=RemoveResponse)
def remove_all_models_endpoint():
    logger.info("Вызван эндпоинт /models/remove_all")
    result = remove_all_models()
    logger.info("Все модели удалены")
    return result
