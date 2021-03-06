# Kaggle MNIST Competition

https://www.kaggle.com/c/digit-recognizer

## Trials

|#|Accuracy|Model|
|-|--|---|
| 1 | 0.99085 | 32 filters  (5, 5) --> max pooling (2, 2) --> 64 fiters (5, 5) --> max pooling (2, 2) --> Dense (512) --> Dense (512) --> Softmax (10) |
| 2 | 0.99114 | Same as 1 and used callback to stop training if we reached certain values
| 3 | 0.99228 | 64 filters  (5, 5) --> max pooling (2, 2) --> 64 fiters (5, 5) --> max pooling (2, 2) --> 64 filters  (5, 5) --> max pooling (2, 2) --> Dense (512) --> Dense (512) --> Softmax (10) |
| 4 | 0.99414 | Same as 3 with no dropout layers and used acc = `0.99992` to stop the training |
| 5 | 0.99300 | same as 4 but with using loss as the stopping criteria and we stopped at loss = `1.1055e-06`. Didn't improve test accuracy.|
| 6 | 0.99071 | same as 5 but stopped at loss = `1.1921e-07`, used 160 epochs, reached the same loss a lot earlier. Didn't improve test accuracy.|
| 9 | 0.99471 | 64 filters  (5, 5) --> batch normalization --> relu --> max pooling (2, 2) --> 128 filters  (5, 5) --> batch normalization --> relu --> max pooling (2, 2) --> 256 filters  (5, 5) --> batch normalization --> relu --> max pooling (2, 2) --> 512 filters  (5, 5) --> batch normalization --> relu --> max pooling (2, 2) --> Dense (512) --> Dropout(0.2) --> Dense (512) --> Dropout(0.2) --> softmax|
| 10 | 0.99371 | Inspired by the Inception model, see [keras_inception.py](keras_incpetion.py) | 
| 11 | 0.99600 | Same as 9 but used ImageDataGenerator to add generated images |
| 12 | 0.99614 | Same as 11 but shear set to 0.1 |
| 13 | <b>0.99685</b> | Same as 12 but added Dropout after every max MaxPooling layer |
| 14 | 0.99685 | Doubled the number of filters in every Conv2D layers |
| 15 | 0.99642 | Double the number of Conv2D layers |
| 16 | 0.99657 | Multi Column Convultion network [keras_mc_cnn.py](keras_mc_cnn.py) | 
| 17 | 0.99685 | Same as 13 but tried the following:<br>- Increate image generation props to 0.15<br>- Adjust the early stopping and lr adjustment to take longer |

## Notes:
1. When increased the number of epochs with no dropout from 60 to 160 the model achieved a higher accuracy (lower loss) but failed to improve the test accuracy.
2. When I used rmsprop instead of adam it didn't improve model accuracy.

