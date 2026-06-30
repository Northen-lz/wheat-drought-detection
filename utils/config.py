from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 模型目录
MODELS_DIR = BASE_DIR / "models"
DET_MODEL_PATH = MODELS_DIR / "detector" / "best.pt"
CLS_MODEL_PATH = MODELS_DIR / "classifier" / "best.onnx"

# 数据目录
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_DIR = DATA_DIR / "wheat_drought_data_upsampled"

# 输出目录
OUTPUTS_DIR = BASE_DIR / "outputs"
CAPTURED_DIR = OUTPUTS_DIR / "captured_photos"
LOG_DIR = OUTPUTS_DIR / "logs"
TRAIN_RESULTS_DIR = OUTPUTS_DIR / "train_results"

# 资源目录
ASSETS_DIR = BASE_DIR / "assets"
BACKGROUND_IMAGE = ASSETS_DIR / "dashboard_bg.jpg"

# 自动创建目录
CAPTURED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# 日志文件
PREDICTION_LOG_PATH = LOG_DIR / "prediction_log.csv"