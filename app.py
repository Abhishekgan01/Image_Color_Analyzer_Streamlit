import streamlit as st
from PIL import Image
import altair as alt
import pandas as pd
import plotly.express as px

from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_image(image_file):
    img = Image.open(image_file)
    return img

def get_image_pixel(filename):
    with Image.open(filename) as rgb_image:
        image_pixel = rgb_image.getpixel((30,30))
    return image_pixel

def main():
    st.title("Image Color Analyzer")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")

        image_file = st.file_uploader("Upload Image", type = ['PNG', 'JPG', 'JPEG'])
        if image_file is not None:
            img = load_image(image_file)
            st.image(img)

            #Analysis
            #Image Pixels
            image_pixel = get_image_pixel(image_file)
            st.write(image_pixel)
    else:
        st.subheader("About")

if __name__ == '__main__':
    main()