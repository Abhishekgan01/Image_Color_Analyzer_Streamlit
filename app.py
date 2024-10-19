import streamlit as st

def main():
    st.title("Image Color Analyzer")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")

        image_file = st.file_uploader("Upload Image", type = ['PNG', 'JPG', 'JPEG'])
        
    else:
        st.subheader("About")

if __name__ == '__main__':
    main()