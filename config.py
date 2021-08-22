from pathlib import Path
from models.modelsconfig import ModelsConfig

class Config:
    """
    Static class containing the following configuration parameters:
        TRAINING_DATASET_PATH
        TEST_DATASET_PATH
        TEXT_LANGUAGE
        SEQUENCE_LENGTH
    """
    TRAINING_DATASET_PATH = Path(__file__).parent / "datasets" / "training_ironita2018.csv"
    TEST_DATASET_PATH = Path(__file__).parent / "datasets" / "test_gold_ironita2018.csv"
    TEXT_LANGUAGE = "italian"
    SEQUENCE_LENGTH = 50
    MODELS_CONFIG = ModelsConfig



if __name__ == '__main__':
    print("Config:")
    print(f"\t{Config.TRAINING_DATASET_PATH=}".replace("Config.", ""))
    print(f"\t{Config.TEST_DATASET_PATH=}".replace("Config.", ""))
    print(f"\t{Config.TEXT_LANGUAGE=}".replace("Config.", ""))
    print(f"\t{Config.SEQUENCE_LENGTH=}".replace("Config.", ""))
