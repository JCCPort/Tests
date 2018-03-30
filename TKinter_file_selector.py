from tkinter import filedialog
from tkinter import *
import pandas as pd

root = Tk()
root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                           filetypes=(("all files", "*.*"), ("jpeg files", "*.jpg")))
data = pd.read_csv('{}'.format(root.filename))
print(data)
