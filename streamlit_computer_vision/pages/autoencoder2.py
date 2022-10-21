from multiprocessing import context
from secrets import choice
from turtle import color, title
import streamlit as st
from tempfile import NamedTemporaryFile
import torch.nn as nn
import torch
from torchvision import io
from PIL import Image
from torchvision import transforms as T
from torchvision import io
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
import os

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder 
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.SELU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.SELU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.SELU()
            )
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        
        #decoder

        self.unpool = nn.MaxUnpool2d(2, 2)
        
        self.conv1_t = nn.Sequential(
            nn.ConvTranspose2d(4, 16, kernel_size=3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.SELU()
            )
        self.conv2_t = nn.Sequential(
            nn.ConvTranspose2d(16, 128, kernel_size=3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.SELU()
            )
        self.conv3_t = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3,padding=1),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
            )

        

    def forward(self, x):
        # encode
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x, indicies = self.pool(x) # ⟸ bottleneck

        # print(x.shape)
        out = x
        # decode
        x = self.unpool(x, indicies)
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        x = self.conv3_t(x)

        return x, out



ae = ConvAutoencoder()
ae.load_state_dict(torch.load('model_denoising_weights.pt', map_location=torch.device('cpu')))

ae.eval()

st.title("Очистка данных от шума")
st.write('Работает на основе автоэнкодера')
count = st.slider('Выберите страницу для очистки от шума:', 215, 216, 1)
path = 'train/'+ str(count)+'.png'
st.image(Image.open(path))
img = io.read_image(path)/255
batched_img = img.unsqueeze(0)
model_out = ae(batched_img)
image = torch.clamp(model_out[0][0][0], 0, 1)

st.image(image.detach().numpy())