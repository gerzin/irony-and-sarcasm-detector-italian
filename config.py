from pathlib import Path
from models.modelsconfig import ModelsConfig

class Config:
    """
    Static class containing the following configuration parameters:
        TRAINING_DATASET_PATH
        TEST_DATASET_PATH
        TEXT_LANGUAGE
        SEQUENCE_LENGTH
        MODELS_CONFIG
        TRAINING_PREPROCESSED_PATH
        TEST_PREPROCESSED_PATH
        VALIDATION_PREPROCESSED_PATH
        TRAIN_SIZE
    """
    TRAINING_DATASET_PATH = Path(__file__).parent / "datasets" / "training_ironita2018.csv"
    TEST_DATASET_PATH = Path(__file__).parent / "datasets" / "test_gold_ironita2018.csv"
    PREPROCESSED_DATASETS_PATH = Path(__file__).parent / "datasets" / "preprocessed"
    TEXT_LANGUAGE = "italian"
    SEQUENCE_LENGTH = 50
    MODELS_CONFIG = ModelsConfig
    TRAINING_PREPROCESSED_PATH = Path(__file__).parent / "datasets" / "preprocessed" / "training.csv"
    TEST_PREPROCESSED_PATH = Path(__file__).parent / "datasets" / "preprocessed" / "test.csv"
    VALIDATION_PREPROCESSED_PATH = Path(__file__).parent / "datasets" / "preprocessed" / "validation.csv"
    TRAIN_SIZE = 0.8


if __name__ == '__main__':
    print("Config:")
    print(f"\t{Config.TRAINING_DATASET_PATH=}".replace("Config.", ""))
    print(f"\t{Config.TEST_DATASET_PATH=}".replace("Config.", ""))
    print(f"\t{Config.PREPROCESSED_DATASETS_PATH=}".replace("Config.", ""))
    print(f"\t{Config.TEXT_LANGUAGE=}".replace("Config.", ""))
    print(f"\t{Config.SEQUENCE_LENGTH=}".replace("Config.", ""))
    print(f"\t{Config.MODELS_CONFIG=}".replace("Config.", ""))
    print(f"\t{Config.TRAINING_PREPROCESSED_PATH=}".replace("Config.", ""))
    print(f"\t{Config.TEST_PREPROCESSED_PATH=}".replace("Config.", ""))
    print(f"\t{Config.VALIDATION_PREPROCESSED_PATH=}".replace("Config.", ""))
    print(f"\t{Config.TRAIN_SIZE=}".replace("Config.", ""))
