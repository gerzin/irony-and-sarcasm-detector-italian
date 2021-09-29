from pathlib import Path


class ModelsConfig:
    SEQUENCE_LENGTH = 50
    BERT_ITA_XXL_CASED = "dbmdz/bert-base-italian-xxl-cased"
    BERT_TOKENIZER_LENGTH = 128
    BERT_CHECKPOINT_DIR = Path(__file__).parent / "pretrained"
    BERT_MODEL_NAME = "bertlstm.h5"


if __name__ == '__main__':
    print("ModelsConfig:")
    print(f"\t{ModelsConfig.SEQUENCE_LENGTH=}".replace("ModelsConfig.", ""))
    print(f"\t{ModelsConfig.BERT_ITA_XXL_CASED=}".replace("ModelsConfig.", ""))
    print(f"\t{ModelsConfig.BERT_TOKENIZER_LENGTH=}".replace("ModelsConfig.", ""))
    print(f"\t{ModelsConfig.BERT_CHECKPOINT_DIR=}".replace("ModelsConfig.", ""))
    print(f"\t{ModelsConfig.BERT_MODEL_NAME=}".replace("ModelsConfig.", ""))


