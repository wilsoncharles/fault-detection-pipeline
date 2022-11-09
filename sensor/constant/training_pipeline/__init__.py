import os
from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME

# definiting common constant variables for full training pipeline
TARGET_COLUMN = 'class'
PIPELINE_NAME:str = "sensor"
ARTIFACT_DIR:str = "artifact"
FILE_NAME = "sensor.csv"

TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = 'model.pkl'
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"


"""
Data Ingestion related constant, starting with DATA_INGESTION variable name
"""
## Edit the comments once the code is written
DATA_INGESTION_COLLECTION_NAME = "car"
DATA_INGESTION_DIR_NAME:str = "data_ingestion" #Data Ingestion main directory
DATA_INGESTION_FEATURE_STORE_DIR:str = "feature_store" ##path inside Data ingestion main directory where the file is extracted from mongo_db and stored in 
DATA_INGESTION_INGESTED_DIR:str = "ingested" #path inside Data ingestion main directory where the train test data is stored in
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION :float = 0.2


"""
Data Valiudation constant, starting with DATA_VALIDATION variable name
"""

DATA_VALIDATION_DIR_NAME : str = 'data_validation'
DATA_VALIDATION_VALID_DIR : str = 'validated'
DATA_VALIDATION_INVALID_DIR : str = 'invalid'
DATA_VALIDATION_DRIFT_REPORT_DIR : str = 'drift_report'
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME :str = 'report.yaml'


"""
Data Transformation constant, starting with DATA_TRANSFORM variable name
"""

DATA_TRANSFORMATION_DIR_NAME : str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR : str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR : str = "transformed_object"

"""
Model trainer related constant, starting with MODEL_TRAINER variable name
"""
MODEL_TRAINER_DIR_NAME : str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR : str = "trainer_model"
MODEL_TRAINER_TRAINED_MODEL_NAME : str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD:float = 0.05