# SMS Spam Classifier
Using the dataset from Kaggle: https://www.kaggle.com/uciml/sms-spam-collection-dataset

## 1. GloVe word vectors + LSTM
In this model I used spacy to tokenize and get the word vectors. word vectors are the input to LSTM and on top of that I used a dense layer of size 1 with sigmoid activation.

Testing accuracy: 0.0.9874  
F1-score: 0.9559

## 2. Embedding + LSTM
Instead of using spacy for tokenization and word vectors, I used keras Tokenizer and Embedding layer. Followed by LSTM and then dense layer.

Test Accuracy: 0.9856  
F1-score: 0.9432