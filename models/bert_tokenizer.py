from models.modelsconfig import ModelsConfig
from transformers import AutoTokenizer
import numpy as np

def get_bert_tokenizer(model_url=ModelsConfig.BERT_ITA_XXL_CASED, tok_len=ModelsConfig.BERT_TOKENIZER_LENGTH):
    """Download the pretrained BERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_url, add_special_tokens=True, max_length=tok_len,
                                              pad_to_max_length=True)

    return tokenizer


def tokenize(sentences, tokenizer):
    """Use the BERT tokenizer to tokenize a list of sentances.
    :returns
        (input_ids, input_masks, input_segments) as np arrays of int32.
    """
    input_ids, input_masks, input_segments = [], [], []
    for sentence in sentences:
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=128, pad_to_max_length=True,
                                       return_attention_mask=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])

    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments,
                                                                                                    dtype='int32')