import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from util import classify, set_background

# Set background
set_background('./bgs/bg5.png')

# Set title
st.title('Pneumonia Classification')

# Set header
st.header('Please upload a chest X-ray image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('./model/harist.h5')

# Load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a.strip() for a in f.readlines()]

# Display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify image
    class_name, conf_score = classify(np.array(image), model, class_names)

    # Write classification
    st.write("## {}".format(class_name))
    st.write("### Score: {}%".format(int(conf_score * 1000) / 10))
