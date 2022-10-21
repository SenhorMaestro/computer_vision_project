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
        # Как работает Conv2dTranspose https://github.com/vdumoulin/conv_arithmetic

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
            nn.SELU()
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


preprocessing = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((250, 500)), # <------ свой размер
        T.ToTensor()
    ]
)

# Function to Read and Manupilate Images
def load_image1(img):
    im = Image.open(img)
    image = np.array(im)
    return im, image

def load_image(image_file):
    convert_tensor = transforms.ToTensor()
    img = Image.open(image_file)
    return convert_tensor(img)
    #return img converted to tensor

def file_save_to_path(image):
    tensor = load_image(image)
    save_image(tensor, '0.png', normalize=True)
    return '0.png'

def generate(image):
    global ae
    ae.eval()
    predictions = []
    #img = read_image(os.path.join('drive/MyDrive/test', i))/255
    img = io.read_image('215.png')/255
    batched_img = img.unsqueeze(0)
    #predictions.append(io.read_image('drive/MyDrive/test/'+i)/255)
    model_out = ae(batched_img)[0].detach().cpu().numpy()
    print(model_out)
    predictions.append(model_out)

    save_image(predictions[-1], '2.jpg', normalize=True)
    return '2.jpg'

# инициализация модели: архитектура + веса
def init_model():
    global ae
    ae = ConvAutoencoder()
    ae.load_state_dict(torch.load('model_denoising_weights.pt', map_location=torch.device('cpu')))
    return ae

def main():
    st.title("Очистка данных от шума")
    st.write('Работает на основе автоэнкодера')
    uploadFile = st.file_uploader(label="Выберите картинку", type=['jpg', 'png'])
    st.image(uploadFile)
    if uploadFile is not None:
        img, _ = load_image(uploadFile)
        st.image(img) 

        img_t = preprocessing(img)  
        st.image(img_t)  
        batch_t = torch.unsqueeze(img_t, 0)
        out = ae(batch_t)
        print(out)

        tensor_image = load_image(uploadFile)
        file_save_to_path(tensor_image)
        st.image(generate(tensor_image))
        #st.image(load_image(generate(uploadFile)))

if __name__ == '__main__':
    init_model()
    main()