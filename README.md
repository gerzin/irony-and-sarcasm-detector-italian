# IronySarcasmDetectorIT
NLP project that analyses Italian tweets and finds out if they are ironic or not, and if they're ironic, if they are sarcastic or not.

### Challenge
The tasks solved in this project are the ones from the [EVALITA 2018](http://www.di.unito.it/~tutreeb/ironita-evalita18/index.html) challenge.
The detailed description can be found here:
    
    @inproceedings{cignarella2018overview,
      title={Overview of the {EVALITA} 2018 task on Irony Detection in {I}talian tweets ({IronITA})},
      author={Cignarella, Alessandra Teresa and Frenda, Simona and Basile, Valerio and Bosco, Cristina and Patti, Viviana and Rosso, Paolo and others},
      booktitle={Sixth Evaluation Campaign of Natural Language Processing and Speech Tools for Italian (EVALITA 2018)},
      volume={2263},
      pages={1--6},
      year={2018},
      organization={CEUR-WS}
    }

## Model
The framework used to develop the models is  tensorflow.
![tensorflow](https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg)

Among all the models we developed, the one producing the best results
has been the BERT-GRU model.

The BERT model we used was a pretrained one available on HuggingFace ([https://huggingface.co/dbmdz/bert-base-italian-xxl-cased](https://huggingface.co/dbmdz/bert-base-italian-xxl-cased)).

![BERTGRUMODEL](https://github.com/gerzin/IronySarcasmDetectorIT/blob/media/.media/bert_gru.jpg)

## Trained BERT-GRU model


[Here](https://mega.nz/file/SaJlnIaR#ujK04KL6z_EKTVNS4K5SaSyhqkW1haKnKH0Xl53pbPQ) you can download the trained **BERT-GRU** model (≈ 500MB).
Another 400MB will be downloaded by the get_bert_tokenizer which will downloaded a pretrained tokenizer
from [HuggingFace.co](https://huggingface.co/)

Here is an example of how to use it:
    
    from models.bert_tokenizer import get_bert_tokenizer, tokenize
    model = keras.models.load_model('path/to/trained/model/location')
    
    tokenizer = get_bert_tokenizer()
    tokenized_tweets = tokenize(tweets, tokenizer)[:-1]
    
    pred = model.predict(tokenized_tweet)
    
    




