from tkinter import Tk, Canvas, Button, Label, Entry, S, E, W
#import pyscreenshot as ImageGrab
from PIL import Image, ImageGrab
import numpy as np
import os

class AppWindow:
	c_height = 364
	c_width = 364
	_bg = 'black'
	_fill = 'white'

	def __init__(self, root):
		root.title('Digit Classifier Training')
		root.resizable(False, False)

		self.c = Canvas(root, width=self.c_width, height=self.c_height + 32)
		self.c.configure(background=self._bg)
		self.c.bind('<B1-Motion>', self.draw)
		self.c.grid(row=0, column=0, columnspan=3, rowspan=4, pady=3)

		_reset = Button(text='Reset', command=self.reset)
		_reset.grid(row=3, column=1, columnspan=2, sticky=S, padx=7, pady=8)
		Button(text='Save', command=lambda: self.save_img(root)).grid(row=3, column=2, sticky=S+E, padx=4, pady=8)
		Button(text='Undo', command=self.undo).grid(row=3, column=0, sticky=S+W, padx=4, pady=8)

		Label(root, text='Digit:').grid(row=4, column=0, sticky=W, padx=4, pady=4)

		vcmd = (Entry().register(self.callback))
		self.entry = Entry(root, width=3, validate='all', validatecommand=(vcmd, '%P'))
		self.entry.grid(row=4, column=0, sticky=W, padx=45, pady=4)

		self.warning = Label(root, text='')
		self.warning.grid(row=4, column=0, columnspan = 3, sticky=W, padx=93)

		self._nr = 0
		if os.path.isdir('./database'):
			self._nr = self.get_nr()
		else:
			os.mkdir(os.getcwd() + '/database/')
			with open('./database/values.txt', 'w'):
				pass

	def undo(self):
		self._nr -= 1
		if self._nr < 0:
			self._nr = 0

		os.remove('database/' + str(self._nr) + '.jpg')

		with open ('database/values.txt', 'r') as f:
			lines = f.readlines()

		lines = lines[:-1]
		with open ('./database/values.txt', 'w') as f:
			for line in lines:
				f.write(line)

		self.warning.config(text='Last entry removed!')

	def callback(self, P):
		if not(str.isdigit(P) or P == ''):
			return False
		return True

	def draw(self, event=None):
		size = 10
		x1, y1 = (event.x - size), (event.y - size)
		x2, y2 = (event.x + size), (event.y + size)

		event.widget.create_oval(x1, y1, x2, y2, fill=self._fill, outline=self._fill)

	def reset(self):
		self.c.delete('all')

	def get_nr(self):
		with open('./database/values.txt', 'r') as f:
			lines = f.readlines()
			nr = len(lines)
		return int(nr)

	def save_img(self, root):
		x1 = root.winfo_rootx() + self.c.winfo_x()
		y1 = root.winfo_rooty() + self.c.winfo_y()
		x2 = x1 + self.c.winfo_width()
		y2 = y1 + self.c.winfo_height() - 32

		img = ImageGrab.grab().crop((x1,y1,x2,y2)).convert('L')
		img = img.resize((28, 28), Image.ANTIALIAS)

		val = self.entry.get()
		if val != '':
			if len(val) > 1:
				self.warning.config(text='Enter only one digit!')
			else:
				img.save('database/' + str(self._nr) + '.jpg')
				with open('database/values.txt', 'a') as f:
					f.write(self.entry.get())
					f.write('\n')
				self._nr += 1
				self.warning.config(text='')
		else:
			self.warning.config(text='No digit entered!')


if __name__ == '__main__':
	root = Tk()
	app = AppWindow(root)
	root.mainloop()