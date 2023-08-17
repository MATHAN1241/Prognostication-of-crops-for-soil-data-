import numpy as np


import pandas as pd
import pickle

import click
from PIL import Image

from time import sleep
#RSA

# STEP 1: Generate Two Large Prime Numbers (p,q) randomly
from random import randrange, getrandbits
from tkinter import *
from tkinter import ttk  
from tkinter import Menu  
from tkinter import messagebox as mbox  
# import filedialog module
from tkinter import filedialog
flg=0;
import tkinter as tk

# Function for opening the
# file explorer window
def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a CSV File",
                                          filetypes = (("CSV files",
                                                        "*.csv*"),
                                                       ("all files",
                                                        "*.*")))
    # Change label contents
    label_file_explorer.configure(text="File Opened: "+filename)
    global f
    f = filename


def start():

    print("Process Started")
    dataset = pd.read_csv(f)
    dataset=dataset.dropna(how="any")
    print(dataset)

    print(dataset.info())

    X = dataset.iloc[:,:].values

    # load the model from disk
    model = pickle.load(open('training_pickle_file', 'rb'))
    ypred = model.predict(X)
    ypred = ypred.round()
    print(ypred)
    app = tk.Tk()
    if(ypred[0]==0):
        print("rice")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is Rice")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is Rice").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
        app.config(menu=menuBar)

    elif(ypred[0]==1):
        print("maize")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is Maize")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is Maize").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
        app.config(menu=menuBar)
    elif(ypred[0]==2):
        print("chickpea")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is chickpea")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is chickpea").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
        app.config(menu=menuBar)
    elif(ypred[0]==3):
        print("kidneybeans")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is kidneybeans")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is kidneybeans").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
        app.config(menu=menuBar)
    elif(ypred[0]==4):
        print("pigeonpeas")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is pigeonpeas")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is pigeonpeas").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
        app.config(menu=menuBar)
    elif(ypred[0]==5):
        print("mothbeans")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is mothbeans")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is mothbeans").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
        app.config(menu=menuBar)
    elif(ypred[0]==6):
        print("mungbean")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is mungbean")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is mungbean").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
        app.config(menu=menuBar)
    elif(ypred[0]==7):
        print("blackgram")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is blackgram")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is blackgram").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
        app.config(menu=menuBar)
    elif(ypred[0]==8):
        print("lentil")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is lentil")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is lentil").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
        app.config(menu=menuBar)
    elif(ypred[0]==9):
        print("pomegranate")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is pomegranate")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is pomegranate").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
        app.config(menu=menuBar)
    elif(ypred[0]==10):
        print("banana")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is banana")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is banana").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
    elif(ypred[0]==11):
        print("mango")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is mango")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is mango").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
    elif(ypred[0]==12):
        print("grapes")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is Grapes")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is Grapes").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
    elif(ypred[0]==13):
        print("watermelon")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is watermelon")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is watermelon").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
    elif(ypred[0]==14):
        print("muskmelon")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is muskmelon")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is muskmelon").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
    elif(ypred[0]==15):
        print("apple")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is Apple")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is Apple").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
    elif(ypred[0]==16):
        print("orange")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is orange")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is orange").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
    elif(ypred[0]==17):
        print("papaya")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is Papaya")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is Papaya").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
    elif(ypred[0]==18):
        print("coconut")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is Coconut")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is Coconut").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
    elif(ypred[0]==19):
        print("cotton")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is Cotton")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is Cotton").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
    elif(ypred[0]==20):
        print("jute")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is Jute")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is Jute").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
    elif(ypred[0]==21):
        print("Coffee")
        label_file_explorer.configure(text="Result for the Data: Predicted crop for cultivation is Coffee")
        app.title("Crop Predicition")
        ttk.Label(app, text="Result for the Data: Predicted crop for cultivation is Coffee").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)

        
if __name__ == '__main__':
    window = Tk()
  
    # Set window title
    window.title('Crop Predicition')
      
    # Set window size
    window.geometry("700x400")
      
    #Set window background color
    window.config(background = "white")
      
    # Create a File Explorer label
    label_file_explorer = Label(window,
                                text = "Please give Input Data",
                                width = 100, height = 4,
                                fg = "blue")
         
    button_explore = Button(window,
                            text = "Browse Crop prediction data file",
                            command = browseFiles)
    button_exit = Button(window,
                         text = "exit",
                         command = exit)  
    button_start = Button(window,
                         text = "Start Analyzing Crop prediction data file",
                         command = start)

       
    # Grid method is chosen for placing
    # the widgets at respective positions
    # in a table like structure by
    # specifying rows and columns
    label_file_explorer.grid(column = 1, row = 1, padx=5, pady=5)
    button_explore.grid(column = 1, row = 3, padx=5, pady=5)
    button_exit.grid(column = 1,row = 9, padx=5, pady=5)
    button_start.grid(column = 1,row = 12, padx=5, pady=5)
      
    # Let the window wait for any events
    
    
    window.mainloop()


    
