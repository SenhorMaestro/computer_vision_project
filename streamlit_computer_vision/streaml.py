import torch
import streamlit as st
import PIL 
from torchvision import transforms
import matplotlib.pyplot as plt
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

st.set_page_config(page_title="My Page Title")

st.header("Распознавание объектов на картинке")
st.write("Загрузите картинку и получите результат детекции")

img_file = st.file_uploader('Выберите картинку', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
if img_file:
    img = PIL.Image.open(img_file)
    results = model(img)
    fig, ax = plt.subplots()
    ax.imshow(results.render()[0])
    plt.axis('off')
    st.pyplot(fig)

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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(28+10, 512, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(128, 28, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(True)
        )

        self.upsample5 = nn.Sequential(
            nn.ConvTranspose2d(28, 1, kernel_size=2, stride=2, padding=2, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, class_vec):
        x = torch.cat([x, class_vec], 1)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)
        return x

def load_image(image_file):
    img = Image.open(image_file)
    # new_height = 200
    # new_width = 200
    # img = img.resize((new_height, new_width), Image.ANTIALIAS)
    return img

def generate(number, c):
    global gen
    gen.eval()
    noise = torch.randn(number, 28, 1, 1).to('cpu')
    c = torch.tensor([c])
    target = nn.functional.one_hot(c.unsqueeze(1).unsqueeze(1).to(torch.int64), 10).permute(0,3,1,2).float().repeat(number, 1, 1, 1)
    tensors = gen(noise, target)
    save_image(tensors, '1.jpg', normalize=True)
    return '1.jpg'

# инициализация модели: архитектура + веса
def init_model():
    global gen
    gen = Generator()
    gen.load_state_dict(torch.load('conditional gan/MNIST_cgan.pt', map_location=torch.device('cpu')))
    return gen

def main():
    st.title("Сгенерирую картинку с числом")
    st.write('Для генерации конкретных картинок использовалась CGAN')
    count = st.slider('Количество цифр:', 1, 10, 1)
    num = st.slider('Число:', 0, 9, 1)
    st.image(load_image(generate(count, num)), width=100*count)