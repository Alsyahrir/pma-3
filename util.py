import base64
from PIL import Image
import numpy as np
import io

def set_background(st, image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        st: Streamlit object
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image_data, model, class_names):
    """
    This function takes raw image data, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image_data (bytes): Raw image data from Streamlit file uploader.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # Convert raw image data to PIL Image
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Resize image to (224, 224)
    image = image.resize((224, 224))

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score
