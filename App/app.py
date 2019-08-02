from tkinter import Tk, Canvas, Button, Label, S, E, W
import pyscreenshot as ImageGrab
from PIL import Image
import numpy as np
from keras.models import load_model

class AppWindow:
    #Parametrii aplicatiei
    c_height = 364 + 35
    c_width = 364
    _model = 'single_models/Best'
    _bg = 'black'
    _fill = 'white'

    def __init__(self, root):
        #Configurarea si initializarea ferestrei
        root.title('Digit Classifier')
        root.resizable(False, False)

        c = Canvas(root, width=self.c_width, height=self.c_height)
        c.configure(background=self._bg)
        c.bind('<B1-Motion>', self.draw)
        c.grid(row=0, column=0, columnspan=3, rowspan=4, pady=3)

        _reset = Button(text='Reset', command = lambda: self.reset(c))
        _reset.grid(row=3, column=1, columnspan=2, sticky=S, padx=7, pady=8)
        _predict = Button(text='Predict', command = lambda: self.make_prediction(root, c, label))
        _predict.grid(row=3, column=2, sticky=S+E, padx=4, pady=8)
        #Button(text='Predict', bg="Gray", fg="Gray85").grid(row=3, column=2, sticky=S+E, padx=4, pady=4)

        label = Label(root, text='Prediction: NaN')
        label.grid(row=4, column=0, sticky=W, padx=4, pady=4)

        self.model = load_model('../best_models/{}.hdf5'.format(self._model))
    
    def draw(self, event=None):
        size = 10
        x1, y1 = (event.x - size), (event.y - size)
        x2, y2 = (event.x + size), (event.y + size)

        event.widget.create_oval(x1, y1, x2, y2, fill=self._fill, outline=self._fill)

    def reset(self, canvas):
        canvas.delete('all')

    def make_prediction(self, root, widget, label):
        label.configure(text = 'Prediction:')

        img = self.get_image(root, widget)
        
        prediction = -1
        result_list = self.model.predict(img)
        best = 999999999
        for i in range(10):
                if abs(1 - result_list[0][i]) < best:
                    best = abs(1 - result_list[0][i])
                    prediction = i
        label.configure(text = 'Prediction: ' + str(prediction))

    def get_image(self, root, widget):
        x1 = root.winfo_rootx() + widget.winfo_x()
        y1 = root.winfo_rooty() + widget.winfo_y()
        x2 = x1 + widget.winfo_width()
        y2 = y1 + widget.winfo_height() - 35

        img = ImageGrab.grab().crop((x1,y1,x2,y2)).convert('L')
        img.save('./big_img.jpg')

        img = img.resize((28,28), Image.ANTIALIAS)
        img.save('./res_.jpg')          #<---- OPTIONAL
        img = np.asarray(img)

        if self._bg == 'white':
            img = 255 - img
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0,2,3,1))
        return img

if __name__ == '__main__':
    root = Tk()
    app = AppWindow(root)
    root.mainloop()