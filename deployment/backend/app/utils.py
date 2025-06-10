import logging
import json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "line_no": record.lineno,
        }
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        return json.dumps(log_record, ensure_ascii=False)


def setup_file_logger():
    logger = logging.getLogger("file-logger")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler("/logs/app.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(JsonFormatter())

    logger.handlers = []
    logger.addHandler(fh)
    return logger
