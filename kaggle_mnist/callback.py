from keras.callbacks import Callback
import warnings

class EarlyStoppingByAccuracy(Callback):
    def __init__(self, value=0.98, verbose=0):
        super().__init__()
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        current = logs.get('acc')
        if current is None:
            warnings.warn("Early stopping requires accuracy available!", RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

class EarlyStoppingByLoss(Callback):
    def __init__(self, value=0.98, verbose=0):
        super().__init__()
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        current = logs.get('loss')
        if current is None:
            warnings.warn("Early stopping requires accuracy available!", RuntimeWarning)

        if current <= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True