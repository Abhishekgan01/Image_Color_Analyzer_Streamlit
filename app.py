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

    else:
        st.subheader("About")

if __name__ == '__main__':
    main()