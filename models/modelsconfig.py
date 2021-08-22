class ModelsConfig:
    SEQUENCE_LENGTH = 50
    BERT_ITA_XXL_CASED = "dbmdz/bert-base-italian-xxl-cased"


if __name__ == '__main__':
    print("ModelsConfig:")
    print(f"\t{ModelsConfig.SEQUENCE_LENGTH=}".replace("ModelsConfig.", ""))
    print(f"\t{ModelsConfig.BERT_ITA_XXL_CASED=}".replace("ModelsConfig.", ""))
