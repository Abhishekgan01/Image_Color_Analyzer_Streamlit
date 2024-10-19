import streamlit as st

def main():
    st.title("Image Color Analyzer")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

if __name__ == '__main__':
    main()