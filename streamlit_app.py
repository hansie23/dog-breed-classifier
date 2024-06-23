import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Constants
IMG_WIDTH, IMG_HEIGHT = 224, 224

labels_df = pd.read_csv("data/labels.csv")

# Load the trained ResNet50 model
@st.cache_resource
def load_resnet50_model():
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    return load_model("model/dog_breed_classifier.h5", options=load_options, compile=False)

def display_model_summary():
    model = load_resnet50_model()
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    st.text(model_summary)

# Function to encode the breed names to numerical labels
@st.cache_data
def encode_breed_labels():
    labels_df = pd.read_csv("data/labels.csv")
    
    label_encoder = LabelEncoder()
    labels_df["breed_label"] = label_encoder.fit_transform(labels_df["breed"])
    # Save the label encoding mapping
    label_mapping = dict(
        zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_)
    )
    return label_mapping

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to predict the class of the image
def predict_image_class(model, img_path):
    labels_df = encode_breed_labels()
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
    confidence = predictions[0][predicted_class]  # Get the confidence score of the predicted class
    class_label = labels_df[predicted_class]    # Get the label of the predicted class
    
    return predicted_class, confidence, class_label

def main():
    st.title("Dog Breed Classifier")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = ["Model Summary", "Total Labels", "Classify Image"]
    choice = st.sidebar.selectbox("Select Option", options)

    if choice == "Model Summary":
        st.write("This is the summary of the Tensorflow ResNet50 model loaded with 'imagenet' weights and fine-tuned for this project.")
        display_model_summary()
        
    elif choice == "Total Labels":
        st.write("This is the total number of labels in the dataset.")
        label_mapping = encode_breed_labels()
        for key, value in label_mapping.items():
            st.write(key, ":", value)
    
    elif choice == "Classify Image":
        st.write("Upload an image of a dog to classify it.")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg"])
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image")
                st.write("")
            
            with col2:
                model = load_resnet50_model()
                predictions = predict_image_class(model, uploaded_file)
                
                st.write("Predicted Class: ", predictions[2])
                st.write("Confidence Score: ", predictions[1])
            
    
if __name__ == "__main__":
    main()