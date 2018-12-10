import matplotlib
matplotlib.use('TkAgg')
from tkinter import *
from tkinter.colorchooser import askcolor
import time 
import numpy as np
import pickle
from preprocess import norm 
import string
import uuid

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self, username, numletter, predict):
        print('HELLO WORLD')

        self.root = Tk()

        print('Initialziing Paint app')

        self.chars = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]    
        # self.chars = ["A", "B"]
        self.index = 0
        self.upper = True
        self.username = username
        self.numLetter = numletter
        self.currentNum = 1
        self.strokeNum = 1
        self.predict = predict

        print('trying to get model')

        self.model = pickle.load(open("model.p", "rb"))

        print('getting other random pen stuff')

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        if self.predict == "c" or self.predict == "C":
            self.currentText_button = Button(self.root, text='A')
        else:
            self.currentText_button = Button(self.root, text=' ')
        self.currentText_button.grid(row=0, column=2)

        self.predictedText_button = Button(self.root, text=' ')
        self.predictedText_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=1, columnspan=5)

        self.currentStroke = []
        self.dataPoints = []
        self.setup()
        self.root.mainloop()

    def onEnter(self, event):
        self.predictedText_button.configure(text=" ")
        self.strokeNum = 1
        quit = False
        self.dataPoints = []
        self.currentStroke = []
        if self.index == len(self.chars) - 1 and self.upper == False and self.currentNum == self.numLetter:
            self.quit()
            quit = True
        if self.currentNum == self.numLetter and self.upper == True:
            self.currentNum = 1
            self.upper = False
        elif self.currentNum == self.numLetter and self.upper == False:
            self.currentNum = 1
            self.upper = True
            self.index += 1
        else:
            self.currentNum += 1
        if not quit:
            self.c.delete("all")
            currentChar = self.chars[self.index]
            if not self.upper:
                currentChar = currentChar.lower()
            if self.predict == "c" or self.predict == "C":
                self.currentText_button.configure(text=currentChar)

    def predictText(self, event):
        chars = sorted(string.ascii_letters)
        predictedLetter = self.model.predict(np.array([norm(np.array(self.dataPoints))]))[0]
        print(predictedLetter)
        sortedPredictedLetter = np.argsort(predictedLetter)
        top3 = sortedPredictedLetter[-3:]
        toDisplay = chars[top3[0]] + " " + str(predictedLetter[top3[0]]) + "\n" + chars[top3[1]] + " " + str(predictedLetter[top3[1]]) + "\n" + chars[top3[2]]  + " " + str(predictedLetter[top3[2]]) 
        print("Got here", toDisplay)
        self.predictedText_button.configure(text=toDisplay)

    def quit(self):
        self.root.destroy()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.root.bind('<Return>', self.onEnter)
        self.root.bind('<Alt_L>', self.predictText)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y
        self.currentStroke.append((self.old_x, self.old_y))

    def saveTxt(self):
        currentChar = self.chars[self.index]
        if self.upper == False:
            currentChar = currentChar.lower()
        fname = 'data4/{3}_{0}_{1:04d}_{2}_{4}.txt'
        np.savetxt(fname.format(currentChar, self.currentNum - 1, self.strokeNum - 1, self.username, uuid.uuid4().hex), np.array(self.currentStroke), delimiter=",")

    def reset(self, event):
        # get boundaries
        x = []
        y = []
        if len(self.currentStroke) > 0:
            for i in range(len(self.currentStroke)):
                x.append(self.currentStroke[i][0])
                y.append(self.currentStroke[i][1])
            x0, x1 = min(x), max(x)
            y0, y1 = min(y), max(y)
            b = .05
            bufferVal = b * max(x1 - x0, y1 - y0)
            x0, y0 = x0 - bufferVal, y0 - bufferVal
            x1, y1 = x1 + bufferVal, y1 + bufferVal
            self.c.create_rectangle(x0, y0, x1, y1, outline="red")
            self.dataPoints.extend(self.currentStroke)
            self.saveTxt()
        self.old_x, self.old_y = None, None
        self.currentStroke = []
        self.strokeNum += 1

if __name__ == '__main__':
    predict = input("Do you want to collect data or predict letters? Type c for collect, p for predict. ")
    numLetter = input("How many of each letter do you want to draw? ")
    name = input("What is your name? ")
    Paint(name, int(numLetter), predict)
