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

def load_image_with_cv(image_file):
    image = Image.open(image_file)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color

def prep_image(raw_img):
    modified_img = cv2.resize(raw_img, (900,600), interpolation=cv2.INTER_AREA)
    modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    return modified_img

def color_analysis(img):
    clf = KMeans(n_clusters=5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys]
    df = pd.DataFrame({'labels':hex_colors, 'Counts':counts.values()})
    return df

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

            myimage = load_image_with_cv(image_file)
            modified_image = prep_image(myimage)
            pix_df = color_analysis(modified_image)

            p01 = px.pie(pix_df, names='labels', values='Counts', color='labels')
            st.plotly_chart(p01)

            col1, col2 = st.columns([1,2])
            with col1:
                st.write(image_pixel)
                st.info("Color Distribution")
                st.write(pix_df)

            with col2:
                p02 = px.bar(pix_df, x='labels', y='Counts', color = 'labels')
                st.plotly_chart(p02)

    else:
        st.subheader("About")

if __name__ == '__main__':
    main()