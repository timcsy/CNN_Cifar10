import tkinter as tk
from tkinter import ttk
import util

class App:
    def __init__(self, root=None):
        root.title('2020 Opencvdl HW1.5')

        mainframe = ttk.Frame(root, padding='8 8 8 8')
        mainframe.grid(row=0, column=0, sticky='wnes')
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        labelframe1 = ttk.Labelframe(mainframe, text='5. Image Processing')
        labelframe1.grid(column=0, row=0, padx=4, pady=4, sticky='nws')

        ttk.Button(labelframe1, text='5.1 Show Train Images', command=util.show_10_images).grid(column=0, row=0, sticky='we')
        ttk.Button(labelframe1, text='5.2 Show Hyperparameters', command=util.show_param).grid(column=0, row=1, sticky='we')
        ttk.Button(labelframe1, text='5.3 Show Model Structure', command=util.show_summary).grid(column=0, row=2, sticky='we')
        ttk.Button(labelframe1, text='5.4 Show Accuracy', command=util.train).grid(column=0, row=3, sticky='we')
        self.index = tk.StringVar()
        self.index.set('(0~9999)')
        ttk.Entry(labelframe1, textvariable=self.index).grid(column=0, row=4, sticky='we')
        ttk.Button(labelframe1, text='5.5 Test', command=test).grid(column=0, row=5, sticky='we')

def test():
    index = int(app.index.get() or 0)
    util.predict(index)

root = tk.Tk()
app = App(root)
root.mainloop()