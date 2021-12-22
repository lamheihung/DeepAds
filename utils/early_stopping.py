import numpy as np
import tensorflow as tf

class EarlyStopping(tf.keras.callbacks.Callback):

    def __init__(self, patience=0, train_metrics='loss', val_metrics='val_loss'):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
         print('epoch = {epoch}, {train_metrics} = {train_metrics_val}, {val_metrics} = {val_metrics_val}'.format(epoch=epoch,
                                                                                                                  train_metrics=self.train_metrics,
                                                                                                                  train_metrics_val=logs[self.train_metrics],
                                                                                                                  val_metrics=self.val_metrics,
                                                                                                                  val_metrics_val=logs[self.val_metrics]))
        current = logs['val_loss']
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))