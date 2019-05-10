import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

rows, cols = 28, 28
_batch_size = 16
_epochs = 30
_drop_rate = 0.4
_optimizer = 'adadelta'
_hidden_layers = 128

def preproc():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 1, rows, cols)
    X_train = np.transpose(X_train, (0,2,3,1))
    X_test = X_test.reshape(X_test.shape[0], 1, rows, cols)
    X_test = np.transpose(X_test, (0,2,3,1))

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test

def create_model():
    classifier = Sequential()

    classifier.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(rows, cols, 1)))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())

    classifier.add(Dense(_hidden_layers, activation='relu', bias_initializer='random_uniform'))
    classifier.add(Dropout(rate=_drop_rate))
    classifier.add(Dense(10, activation='softmax'))

    classifier.compile(optimizer=_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

def train(X_train, y_train, X_test, y_test):
    model = create_model()

    checkpointer = ModelCheckpoint('./best_models/single_models/Conv128.hdf5',
                                   monitor='val_acc', mode='max', save_best_only=True, verbose=1)

    history = model.fit(X_train, y_train,
                        batch_size=_batch_size,
                        epochs=_epochs,
                        callbacks=[checkpointer],
                        validation_data=(X_test, y_test))

    return history

def save(val_loss, val_acc, loss, acc):
    with open('./best_models/test_single_models/Conv128.txt', 'w') as f:
        f.write('val_loss: {}\nval_acc: {}\nloss: {}\nacc: {}'
                .format(val_loss, val_acc, loss, acc))


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = preproc()
    history = train(X_train, y_train, X_test, y_test)

    val_loss = history.history.get('val_loss')
    val_acc = history.history.get('val_acc')
    loss = history.history.get('loss')
    acc = history.history.get('acc')
    save(val_loss, val_acc, loss, acc)