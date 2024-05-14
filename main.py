import streamlit as st
from PIL import Image

def main():
    st.title('Image Upload and Display Example')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Image successfully uploaded and displayed below.")

if __name__ == "__main__":
    main()
