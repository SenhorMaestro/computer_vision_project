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
# print("Sreamlit version", st.__version__)


def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    st.title("Пицца или не пицца?")
    st.write('Для классификации пицц/не пицц использовалась предобученная сеть **ResNet18**')
    image_file = st.file_uploader("Загрузите изображение", type=["png","jpg","jpeg"])
    if image_file is not None:
    #     file_details = {"filename":image_file.name, "filetype":image_file.type,
    #                     "filesize":image_file.size}
    #     st.write(file_details)

        # To View Uploaded Image
        # resize = T.Resize((224, 224))
        # print('Type':, type(image_file))
        dictionar_class = {
            0: 'не пицца',
            1: 'пицца'}

        decode = lambda x: dictionar_class[int(x)] 
        convert_tensor = transforms.ToTensor()

        resize = T.Resize((224, 224))
        img = resize(convert_tensor(load_image(image_file)))
        print(type(img))

        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18")
        model.fc = nn.Linear(512, 1)
        model.load_state_dict(torch.load('model_pizza.pt', map_location=torch.device('cpu')))

        model.eval()
        # plt.imshow(torch.permute(img, (1, 2, 0)))
        label = decode(torch.round(model(img.unsqueeze(0)).sigmoid()).item())
        st.title('Возможно, это ' + label)
        st.image(load_image(image_file), width=250)
        # st.title(label)
        

if __name__ == "__main__":
    main()