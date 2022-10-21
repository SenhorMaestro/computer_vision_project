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
