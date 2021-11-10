from pathlib import Path


class ModelsConfig:
    SEQUENCE_LENGTH = 50
    BERT_ITA_XXL_CASED = "dbmdz/bert-base-italian-xxl-cased"
    BERT_TOKENIZER_LENGTH = 128
    BERT_CHECKPOINT_DIR = Path(__file__).parent / "pretrained"
    BERT_MODEL_NAME = "bertlstm.h5"


if __name__ == '__main__':
    def print_(x: str):
        if type(x) is not str:
            raise TypeError("this version of print only accepts strings")
        print(x.replace("Config.", ""))
    print("ModelsConfig:")
    print_(f"\t{ModelsConfig.SEQUENCE_LENGTH=}")
    print_(f"\t{ModelsConfig.BERT_ITA_XXL_CASED=}")
    print_(f"\t{ModelsConfig.BERT_TOKENIZER_LENGTH=}")
    print_(f"\t{ModelsConfig.BERT_CHECKPOINT_DIR=}")
    print_(f"\t{ModelsConfig.BERT_MODEL_NAME=}")


