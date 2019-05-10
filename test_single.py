from keras.datasets import mnist
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

def std():
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        z_test = X_train[0]
        test = X_train[6]
        test = np.expand_dims(test, axis=0)
        test = np.expand_dims(test, axis=0)
        test = np.transpose(test, (0,2,3,1))

        model = load_model('./best_models/128-128-0.5.hdf5')
        result = model.predict(test) == True

        for i in range(10):
                if result[0][i] == True:
                        nr=i
                        break
        
                print(nr)

def invert_grayscale(img, size=(28,28)):
        for i in range(size[0]):
            for j in range(size[1]):
               img[i][j] = 255 - img[i][j]

        return img

def test():
        img = image.load_img('./App/img.jpg', target_size=(28, 28), color_mode='grayscale')
        img = image.img_to_array(img)
        img = invert_grayscale(img)
        img = np.expand_dims(img, axis=0)

        model = load_model('./best_models/128-128-0.5.hdf5')
        result_list = model.predict(img)
        best = 999999999
        for i in range(10):
                if abs(1 - result_list[0][i]) < best:
                        best = abs(1 - result_list[0][i])
                        result = i
        print(result)

if __name__ == '__main__':
        #std()
        test()
