from pathlib import Path


class Config:
    """
    Static class containing the following configuration parameters:
        TRAINING_DATASET_PATH
    """
    TRAINING_DATASET_PATH = Path(__file__).parent / "datasets" / "training_ironita2018.csv"


if __name__ == '__main__':
    print(f"{Config.TRAINING_DATASET_PATH=}".replace("Config.", ""))
