#import the libraries that are required
import streamlit as st
import datetime
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import streamlit_theme as stt
import io
import cv2
from app_utils import get_model_bin
import os
from app_utils import convertToJPG
from app_utils import generate_random_filename
from app_utils import download
import requests
from app_utils import clean_all
from app_utils import create_directory

from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import convertToJPG
import urllib
from os import path
import torch

import fastai
from deoldify.visualize import *
from pathlib import Path
import traceback



#---------------------------------Page title------------------------------------------

st.set_page_config(page_title="Post Graduate Certification in Applied AI")

col1, mid, col2, col3 = st.beta_columns([1, 1, 4, 1])
with col1:
    st.image('wiley_nxt.jfif', width = 100)
with col2:
    st.subheader("Post Graduate Certification in Applied AI")
with col3:
    st.image('iit_mandi.jfif', width = 100)




EXTERNAL_DEPENDENCIES = {
"ColorizeArtistic_gen.pth": {
"url": "https://drive.google.com/file/d/1il525cE6qq2x-uU-YLw12suHhs6bPCCt/view?usp=sharing",
"size": 249165}
}


def resize_one(fn, img_size=800):
    dest = 'image/image_bw/temp_bw.jpg'

    # Load the image
    img = cv2.imread(str(fn))
    height, width = img.shape[0], img.shape[1]
    if max(width, height) > img_size:
        if height > width:
            width = width * (img_size / height)
            height = img_size
            img = cv2.resize(img, (int(width), int(height)))
        elif height <= width:
            height = height * (img_size / width)
            width = img_size
            img = cv2.resize(img, (int(width), int(height)))
    cv2.imwrite(str(dest), img)

def create_learner(path,file):
   learn_gen=load_learner(path,file)
   return learn_gen

def main():
   # for filename in EXTERNAL_DEPENDENCIES.keys():
   #    download_file(filename)
   st.title("Black&White Photos Colorisation")
   uploaded_file = st.file_uploader("upload a black&white photo", type=['jpg', 'png', 'jpeg', 'jfif'])

   if uploaded_file is not None:
       g = io.BytesIO(uploaded_file.read())  # BytesIO Object
       temporary_location = "image/temp.jpg"
       file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
       st.write(file_details)
       with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
           out.write(g.read())  # Read bytes into file
           # close file
           out.close()
   resize_one("image/temp.jpg", img_size=800)
   st.image("image/temp.jpg", width=800)

   start_analyse_file = st.button('Analyse uploaded file')
   if start_analyse_file == True:
       learn_gen = create_learner(path='', file='ColorizeArtistic_gen.pth')
       predict_img("image/image_bw/temp_bw.jpg", learn_gen, img_width=800)

if __name__ == "__main__":
   main()

def predict_img(fn,learn_gen,img_width=640):
   _,img,b=learn_gen.predict(open_image(fn))
   img_np=image2np(img)
   st.image(img_np,clamp=True,width=img_width)


st.sidebar.markdown("""
[Project Architeture](https://drive.google.com/file/d/1CvPsTD9EYfvhREMhK1GsoOqIZxVb0dfT/view)""")
st.sidebar.markdown("""
[Reasearch Paper](https://drive.google.com/file/d/1lf7GRaAmg5lwJF3Ik7irdFncvgc64nzo/view?usp=sharing)""")
st.sidebar.markdown("""
[Presentation](https://docs.google.com/presentation/d/15HfriKFJ5acUQJ1qqCTX2-4JDUT5InTOfC6PH3KF9EE/edit?usp=sharing)""")

st.sidebar.subheader("Presented by:")
st.sidebar.write("1. Deepika Goel")
st.sidebar.write("2. Rahul Sharma")
st.sidebar.write("3. Skanda")
st.sidebar.write("4. Jay Gandhi")

st.sidebar.subheader("Under Guidance Of Mentors:")
st.sidebar.write("1. Ranjith")
st.sidebar.write("2. Anoushka")

st.sidebar.subheader("Under Guidance Of Professors:")
st.sidebar.write("1. Aditya")
st.sidebar.write("2. Arnav")

