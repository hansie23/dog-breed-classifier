import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

IMG_WIDTH, IMG_HEIGHT = 224, 224
labels_df = pd.read_csv("data/labels.csv")

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

@st.cache_data
def encode_breed_labels():
    labels_df = pd.read_csv("data/labels.csv")
    label_encoder = LabelEncoder()
    labels_df["breed_label"] = label_encoder.fit_transform(labels_df["breed"])
    label_mapping = dict(
        zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_)
    )
    return label_mapping

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_image_class(model, img_path):
    labels_df = encode_breed_labels()
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]
    class_label = labels_df[predicted_class]
    
    return predicted_class, confidence, class_label

def main():
    st.title("Dog Breed Classifier")
    
    # sidebar for navigation
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
                
                st.write(f"Predicted Class: **<span style='color:green'>{predictions[2]}</span>**", unsafe_allow_html=True)
                # st.write(f"Confidence Score: **<span style='color:green'>{str(predictions[1])}</span>**", unsafe_allow_html=True)
            
    
if __name__ == "__main__":
    main()
    