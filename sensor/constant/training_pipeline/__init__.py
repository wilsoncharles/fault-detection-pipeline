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
