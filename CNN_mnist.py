import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#from sklearn.preprocessing import OneHotEncoder

#Lista cu diferiti parametri pentru reteaua neuronala
rows, cols = 28, 28
_batch_size = [32, 64, 128]
_epochs = 10
_droprate = [0.2, 0.4, 0.5]
_optimizer = 'adadelta'
_hidden_layers = [64, 128, 256, 512]


def preproc():
    '''
    Importarea bazei de date initiale din biblioteca Keras
    si preprocesarea imaginilor
    '''

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 1, rows, cols)
    X_train = np.transpose(X_train, (0,2,3,1))
    X_test = X_test.reshape(X_test.shape[0], 1, rows, cols)
    X_test = np.transpose(X_test, (0,2,3,1))
    
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    '''
    y_train = y_train.reshape(-1, 1)
    y_train = OneHotEncoder().fit_transform(y_train).toarray()
    y_test = y_test.reshape(-1, 1)
    y_test = OneHotEncoder().fit_transform(y_test).toarray()
    '''

    return X_train, y_train, X_test, y_test


def model(_dr, _opt, _hl):
    '''
    Crearea modelului modificand, pe rand, parametrii acestuia
    '''

    classifier = Sequential()

    classifier.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(rows, cols, 1)))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())

    classifier.add(Dense(_hl, activation='relu', bias_initializer='random_uniform'))
    classifier.add(Dropout(rate=_dr))
    classifier.add(Dense(10, activation='softmax'))

    classifier.compile(optimizer = _opt, loss = 'categorical_crossentropy', metrics=['accuracy'])
    return classifier


def train(_bs, _ep, _dr, _opt, _hl):
    '''
    Antrenarea modelelor si salvarea celor mai bune dintre acestea
    in folderul best_models/
    '''
    X_train, y_train, X_test, y_test = preproc()
    classifier = model(_dr, _opt, _hl)

    checkpointer = ModelCheckpoint('./best_models/{}-{}-{}.hdf5'.format(_bs, _hl, _dr), 
                                   monitor='val_acc', mode='max', save_best_only=True, verbose=1)
    history = classifier.fit(X_train, y_train,
                             batch_size=_bs,
                             epochs=_ep,
                             callbacks=[checkpointer],
                             validation_data=(X_test, y_test))

    return history


def save(_bs, _dr, _hl, val_loss, val_acc, loss, acc):
    '''
    Salvarea caracteristicilor celor mai bune modele in folderul
    best_models/ pentru a putea fi usor reprezentate grafic
    '''
    with open('./best_models/{}-{}-{}.txt'.format(_bs, _hl, _dr), 'w') as f:
        f.write('val_loss: {}\nval_acc: {}\nloss: {}\nacc: {}'
                .format(val_loss, val_acc, loss, acc))

if __name__ == '__main__':
    '''
    Parcurgerea si antrenarea tuturor modelelor
    Gasirea unui model optim
    '''

    b_val_acc = -1.0

    for _bs in _batch_size:
        for _hl in _hidden_layers:
            for _dr in _droprate:
                history = train(_bs, _epochs, _dr, _optimizer, _hl)

                val_loss = history.history.get('val_loss')
                val_acc = history.history.get('val_acc')
                loss = history.history.get('loss')
                acc = history.history.get('acc')
                save(_bs, _dr, _hl, val_loss, val_acc, loss, acc)

                if val_acc[-1] >= b_val_acc:
                    b_val_acc = val_acc[-1]
                    print('\nBetter ACCURACY found! Writing to file....\n')
                    with open('./best_models/zzz_Best_Results.txt', 'w') as f:
                        f.write('Batch Size: {}\nNumber of Hiden Layers: {}\nDrop Rate: {}\nAccuracy: {}%'
                               .format(_bs, _hl, _dr, val_acc[-1]*100))

