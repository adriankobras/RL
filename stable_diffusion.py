import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from auth_token import AuthToken

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app
app = tk.Tk()
app.geometry("532x622")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

app.mainloop()