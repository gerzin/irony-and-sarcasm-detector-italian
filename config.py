from pathlib import Path


class Config:
    """
    Static class containing the following configuration parameters:
        TRAINING_DATASET_PATH
        TEXT_LANGUAGE
    """
    TRAINING_DATASET_PATH = Path(__file__).parent / "datasets" / "training_ironita2018.csv"
    TEXT_LANGUAGE = "italian"


if __name__ == '__main__':
    print(f"{Config.TRAINING_DATASET_PATH=}".replace("Config.", ""))
    print(f"{Config.TEXT_LANGUAGE=}".replace("Config.", ""))
