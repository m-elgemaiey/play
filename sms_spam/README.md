# SMS Spam Classifier
Using the dataset from Kaggle: https://www.kaggle.com/uciml/sms-spam-collection-dataset

## 1. GloVe word vectors + LSTM
In this model I used spacy to tokenize and get the word vectors. word vectors are the input to LSTM and on top of that I used a dense layer of size 1 with sigmoid activation.

Testing accuracy: 0.9847  
F1-score: 0.9473

## 2. Embedding + LSTM
Instead of using spacy for tokenization and word vectors, I used keras Tokenizer and Embedding layer. Followed by LSTM and then dense layer.

Test Accuracy: 0.9856  
F1-score: 0.9432

## 3. Combined: GloVe + Embedding
Concatenate the models in 1 and 2

Test Accuracy: 0.9865   
F1-score: 0.9466

## 4. Convolution model
Using GloVe vectors and Conv1D layers

Test Accuracy: 0.9847   
F1-score: 0.9463
