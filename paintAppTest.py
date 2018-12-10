from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Line

global dataPoints
global username
global character
global index

class DrawInput(Widget):
    
    def on_touch_down(self, touch):
        print(touch)
        with self.canvas:
            touch.ud["line"] = Line(points=(touch.x, touch.y))
        
    def on_touch_move(self, touch):
        print(touch)
        touch.ud["line"].points += (touch.x, touch.y)
        dataPoints.append("(" + str(touch.x) + "," + str(touch.y) + ")")
		
    def on_touch_up(self, touch):
        print("RELEASED!",touch)

class SimpleKivy4(App):
    
    def build(self):
        return DrawInput()

    def on_stop(self, *args):
        fileName = username + character + str(index) + ".txt"
        f = open(fileName, "w+")
        for i in range(len(dataPoints)):
            f.write(dataPoints[i])
        f.close()
        return True

if __name__ == "__main__":
    numLetter = input("How many of each letter do you want to draw? ")
    username = input("What is your name? ")
    chars = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]    
    for char in chars:
        character = char
        index = 1
        for i in range(int(numLetter)):
            dataPoints = []
            SimpleKivy4().run()
            index += 1
