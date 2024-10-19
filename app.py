import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px

from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors as mcolors
import numpy as np
import cv2


def load_image(image_file):
    img = Image.open(image_file)
    return img


def get_image_pixel(filename):
    with Image.open(filename) as rgb_image:
        image_pixel = rgb_image.getpixel((30, 30))
    return image_pixel


def load_image_with_cv(image_file):
    image = Image.open(image_file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def rgb_to_color_name(rgb_color):
    css4_colors = mcolors.CSS4_COLORS
    min_dist = float("inf")
    closest_color_name = None

    for name, hex_code in css4_colors.items():
        r, g, b = mcolors.hex2color(hex_code)
        r, g, b = [int(255 * val) for val in [r, g, b]]  # Convert to 0-255 range
        dist = np.sqrt((rgb_color[0] - r) ** 2 + (rgb_color[1] - g) ** 2 + (rgb_color[2] - b) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_color_name = name

    return closest_color_name


def rgb_to_hex(rgb_color):
    """ Convert RGB color to Hex format """
    return mcolors.to_hex([val / 255 for val in rgb_color])


def prep_image(raw_img):
    modified_img = cv2.resize(raw_img, (900, 600), interpolation=cv2.INTER_AREA)
    modified_img = modified_img.reshape(modified_img.shape[0] * modified_img.shape[1], 3)
    return modified_img


def color_analysis(img):
    clf = KMeans(n_clusters=5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    color_names = [rgb_to_color_name(ordered_colors[i]) for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    
    # Create a DataFrame with color names and counts
    df = pd.DataFrame({'labels': color_names, 'Counts': counts.values(), 'hex': hex_colors})
    return df


def main():
    st.title("Image Color Analyzer")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")

        image_file = st.file_uploader("Upload Image", type=['PNG', 'JPG', 'JPEG'])
        if image_file is not None:
            img = load_image(image_file)
            st.image(img)

            # Image Pixels
            image_pixel = get_image_pixel(image_file)
            st.write(f"Pixel Color at (30, 30): {image_pixel}")

            myimage = load_image_with_cv(image_file)
            modified_image = prep_image(myimage)
            pix_df = color_analysis(modified_image)

            # Map color labels to hex colors for consistency in Plotly
            color_discrete_map = {row['labels']: row['hex'] for _, row in pix_df.iterrows()}

            # Plot pie chart
            p01 = px.pie(pix_df, names='labels', values='Counts', color='labels',
                         color_discrete_map=color_discrete_map)
            st.plotly_chart(p01)

            # Plot bar chart
            col1, col2 = st.columns([1, 2])
            with col1:
                st.info("Color Distribution")
                st.write(pix_df)

            with col2:
                p02 = px.bar(pix_df, x='labels', y='Counts', color='labels',
                             color_discrete_map=color_discrete_map)
                st.plotly_chart(p02)

            # New Graphs

            # Line Chart
            st.subheader("Line Chart of Color Distribution")
            line_fig = px.line(pix_df, x='labels', y='Counts', title="Line Chart")
            st.plotly_chart(line_fig)

            # 3D Scatter Plot
            st.subheader("3D Scatter Plot of Color Clusters")
            # To demonstrate, we're plotting labels, counts, and index as z-axis.
            scatter_3d_fig = px.scatter_3d(pix_df, x='labels', y='Counts', z=pix_df.index, color='labels',
                                           title="3D Scatter Plot", color_discrete_map=color_discrete_map)
            st.plotly_chart(scatter_3d_fig)

            # Histogram
            st.subheader("Histogram of Color Counts")
            hist_fig = px.histogram(pix_df, x='labels', y='Counts', color='labels',
                                    title="Histogram", color_discrete_map=color_discrete_map)
            st.plotly_chart(hist_fig)

            # Bubble Chart
            st.subheader("Bubble Chart of Color Counts")
            bubble_fig = px.scatter(pix_df, x='labels', y='Counts', size='Counts', color='labels',
                                    title="Bubble Chart", color_discrete_map=color_discrete_map)
            st.plotly_chart(bubble_fig)

            # Heatmap
            st.subheader("Heatmap of Color Labels and Counts")
            heatmap_data = pix_df.pivot_table(index='labels', values='Counts', aggfunc='sum').reset_index()
            heatmap_fig = px.imshow(heatmap_data[['Counts']], labels=dict(x="Labels", y="Counts"),
                                    x=heatmap_data['labels'], title="Heatmap")
            st.plotly_chart(heatmap_fig)

    else:
        st.subheader("About")


if __name__ == '__main__':
    main()
