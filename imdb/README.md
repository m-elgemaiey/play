# IMDB sentiment classification dataset.

https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb

## Trials

|#|Accuracy|Model|
|-|--|---|
| 1 | 0.83 | Embedding(128) -> LSTM(128) -> Dense(64) -> Dense(64) |
| 2 | 0.83 | Embedding(64)  -> LSTM(32)  -> Dense(64) -> Dense(64) |
