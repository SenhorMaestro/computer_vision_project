from torchvision import models
import torch
import streamlit as st
from PIL import Image
import numpy as np

alexnet = models.alexnet(weights='IMAGENET1K_V1')

from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

with open('imagenet1000_clsidx_to_labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]

def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return im, image

st.header("Классификация произвольного изображения на модели AlexNet")
st.write("Загрузите картинку и получите результат классификации")

uploadFile = st.file_uploader(label="Выберите картинку", type=['jpg', 'png'])
if uploadFile is not None:
    img, _ = load_image(uploadFile)
    st.image(img)    

    img_t = transform(img)    
    batch_t = torch.unsqueeze(img_t, 0)

    alexnet.eval()

    out = alexnet(batch_t)
    
    _, index = torch.max(out, 1) 
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    st.title('Предсказание модели:')
    if round(percentage[index[0]].item())<50:
        st.metric(label="Вероятно, это", value=labels[index[0]].split("'")[1], delta=percentage[index[0]].item()*(-1))
    else:
        st.metric(label="Должно быть, это", value=labels[index[0]].split("'")[1], delta=percentage[index[0]].item())
