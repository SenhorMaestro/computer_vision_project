import torch
import streamlit as st
import PIL 
from torchvision import transforms
# Устанавливаем логгер для детектрона
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Импорты
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.engine import DefaultTrainer
yaml_path = 'COCO-Detection/faster_rcnn_R_50_C4_1x.yaml'

cfg = get_cfg() 
cfg.merge_from_file(model_zoo.get_config_file(yaml_path))

# ## Проверяем, существует ли папка для сохранения обученной модели
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Подгружаем обученную модель
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # путь к обученной модели
# устанавливаем порог обнаружения
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  

# создаем объект для построения предсказаний
predictor = DefaultPredictor(cfg)








# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
results = model(imgs)
#print(results)

# Results
#results.print()
#results.save()  # or .show()
print(type(results.ims[0]))

def take_picture(image):
    filename = image
    results = model(image)
    #results.show()
    results.save()
    #return results.ims[0]
    pil_image = transforms.ToPILImage()(results.ims[0])
    return torch.permute(pil_image, (1,2,0))


st.header("Generate ASCII images using GAN")
st.write("Choose any image and get corresponding ASCII art:")

uploaded_file = st.file_uploader("Choose an image...", type=['png','jpg','jpeg'])

#open the input file
#img = PIL.Image.open(uploaded_file)

if uploaded_file is not None:
    #src_image = load_image(uploaded_file)
    image = PIL.Image.open(uploaded_file)	
	
    st.image(uploaded_file, caption='Input Image', use_column_width=True)
    #st.write(os.listdir())
    im = take_picture(uploaded_file)
    im1 = PIL.Image.fromarray(im)
    im2 = PIL.Image.open(im)	
    st.image(im, caption='ASCII art', use_column_width=True)