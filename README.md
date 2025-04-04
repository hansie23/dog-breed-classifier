# Dog Breed Classifier

## Description

This project implements a dog breed classifier using a deep learning model. It features a Streamlit web application that allows users to upload an image of a dog and get a prediction of its breed.

## Features

* **Image Classification**: Upload a JPG/JPEG image of a dog to predict its breed.
* **Model Summary**: View the architecture summary of the underlying deep learning model.
* **Label Information**: Display the list of dog breeds the model is trained to recognize.

## Model

The classification is performed using a pre-trained ResNet50 model, which has been fine-tuned for the task of dog breed identification.

## Installation

1.  **Clone or download the repository:**
    ```bash
    git clone <repository-url>
    cd dog-breed-classifier
    ```
2.  **Install dependencies:**
    Make sure you have Python installed. Then, install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    The required libraries are:
    * pandas
    * numpy
    * matplotlib
    * scikit-learn
    * tensorflow

## Usage

To run the Streamlit application, navigate to the project directory in your terminal and run the following command:

```bash
streamlit run app.py