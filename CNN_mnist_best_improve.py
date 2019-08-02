from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

#Adresa celui mai bun model
INITIAL_MODEL = 'single_models/Conv128'
NAME = 'test'

def train():
    #Incarcarea modelului optim
    classifier = load_model('./best_models/{}.hdf5'.format(INITIAL_MODEL))

    #Incarcarea in memorie si preprocesarea imaginilor din baza de date secundara
    train_datagen = ImageDataGenerator(rotation_range=5)
    test_datagen = ImageDataGenerator(rotation_range=5)

    train_generator = train_datagen.flow_from_directory('MNIST_data/imporve_db/training_set',
                                                        target_size=(28, 28),
                                                        color_mode='grayscale',
                                                        batch_size=25,
                                                        class_mode='categorical',
                                                        shuffle=True)

    test_generator = test_datagen.flow_from_directory('MNIST_data/imporve_db/test_set',
                                                      target_size=(28, 28),
                                                      color_mode='grayscale',
                                                      batch_size=1,
                                                      class_mode='categorical')

    checkpionter = ModelCheckpoint('./best_models/single_models/{}.hdf5'.format(NAME),
                                   monitor='val_acc', mode='max', save_best_only=True, verbose=1)

    #Reantrenarea modelului
    history = classifier.fit_generator(train_generator,
                                       steps_per_epoch=1500,
                                       epochs=20,
                                       validation_data=test_generator,
                                       validation_steps=362,
                                       callbacks=[checkpionter],
                                       use_multiprocessing=True)
                                    
    return history

def save(val_loss, val_acc, loss, acc):
    with open('./best_models/single_models/{}.txt'.format(NAME), 'w') as f:
        f.write('val_loss: {}\nval_acc: {}\nloss: {}\nacc: {}'
                .format(val_loss, val_acc, loss, acc))

if __name__ == '__main__':
    history = train()

    val_loss = history.history.get('val_loss')
    val_acc = history.history.get('val_acc')
    loss = history.history.get('loss')
    acc = history.history.get('acc')
    save(val_loss, val_acc, loss, acc)