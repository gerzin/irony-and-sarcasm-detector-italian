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

## Challenge Results

 * Task A

| Name             	| F1 Avg    	|
|------------------	|-----------	|
| ItaliaNLP        	| 0.731     	|
| ItaliaNLP        	| 0.713     	|
| **Our Model**     	| **0.712** 	|
| UNIBA            	| 0.710     	|
| UNIBA            	| 0.710     	|
| X2Check          	| 0.704     	|
| UNITOR           	| 0.700     	|
| UNITOR           	| 0.700     	|
| X2Check          	| 0.695     	|
| Aspie96          	| 0.695     	|
| X2Check          	| 0.693     	|
| X2Check          	| 0.683     	|
| UOIRO            	| 0.651     	|
| UOIRO            	| 0.646     	|
| UOIRO            	| 0.629     	|
| UOIRO            	| 0.614     	|
| `baseline-random` |    `0.505`    |
| venses-itgetarun 	| 0.470     	|
| venses-itgetarun 	| 0.420     	|
| baseline-mfc     	| 0.33      	|

* Task B

| Name             	| F1 Avg 	|
|------------------	|--------	|
| **Our Model**    	| **0.536** |
| UNITOR           	| 0.520  	|
| UNITOR           	| 0.518  	|
| ItaliaNLP        	| 0.516  	|
| ItaliaNLP        	| 0.503  	|
| Aspie96          	| 0.465  	|
| `baseline-random` | `0.337`  	|
| venses-itgetarun 	| 0.236  	|
| baseline-mfc     	| 0.223  	|
| venses-itgetarun 	| 0.199  	|

    
### Code structure
The main code is split across there three folders:
* __preprocessing__ - contains utilities for text preprocessing implementing the pipeline through which the tweets will pass.
* __models__ - contains the models implementations.
* __notebooks__ - contains the Jupyter notebooks we used to develop and test the code. The notebooks containing the models' implementations are meant to be run on Colab.




