import torch
import streamlit as st
import PIL 
from torchvision import transforms
import matplotlib.pyplot as plt 

model = torch.hub.load('ultralytics/yolov5', 'custom',path='cv_project/res_models/yolov5/best.pt')
img_file = st.file_uploader('Choose file', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
if img_file:
    img = PIL.Image.open(img_file)
    results = model(img)
    fig, ax = plt.subplots()
    ax.imshow(results.render()[0])
    st.pyplot(fig)