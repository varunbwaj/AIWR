import pandas as pd
import os
import random
import streamlit as st
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
model = load('model.joblib')

# Add a title and some text to the app
st.title("FoodieFinder: A Comprehensive Culinary Keyword Dataset")
st.write("Enter a keyword and we'll predict the top 5 dishes related to it!")

# Create a text input for the user
user_input = st.text_input("Enter your keyword here")

probabilities = None

# When the user enters text, transform it and make a prediction
if user_input:
    # Transform the user input
    user_input_transformed = model.named_steps['tfidfvectorizer'].transform([user_input])


    # Get probabilities
    probabilities = model.named_steps['svc'].predict_proba(user_input_transformed)

    # Get the indices of the top 5 predictions
    top5_indices = probabilities[0].argsort()[-5:][::-1]

    # Get the class labels
    class_labels = model.named_steps['svc'].classes_

    # Display the most probable dish
    st.markdown("### Most probable dish:")
    st.write(class_labels[top5_indices[0]])

    # Get a list of all images in the directory for this dish
    image_dir = os.path.join('test', class_labels[top5_indices[0]])
    images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]

    # Randomly select up to 5 images and display them
    selected_images = random.sample(images, min(5, len(images)))

    # Create columns for the images
    cols = st.columns(len(selected_images))

    for col, image in zip(cols, selected_images):
        col.image(image)

    # Display the top 4 predicted dishes
    st.markdown("### Not the dish in mind? Here are some other similar dishes:")
    # st.markdown("### Top 4 predicted dishes:")
    for i in top5_indices[1:]:
        dish = class_labels[i]
        st.write(dish)

        # Get a list of all images in the directory for this dish
        image_dir = os.path.join('test', dish)
        images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]

        # Randomly select up to 5 images and display them
        selected_images = random.sample(images, min(5, len(images)))

        # Create columns for the images
        cols = st.columns(len(selected_images))

        for col, image in zip(cols, selected_images):
            col.image(image)


    # Get the top 5 probabilities
    probs = sorted(probabilities[0], reverse=True)[:5]

    # Create a list to store the classes and their probabilities
    data = []

    # Iterate over the top 5 indices
    for i, idx in enumerate(top5_indices):
        # Append the class and its probability to the data list
        data.append([class_labels[idx], probs[i]])

    st.markdown("### Stats for nerds:")

    # Create a DataFrame from the data list
    df = pd.DataFrame(data, columns=['Class', 'Probability'],index=[1,2,3,4,5])

    # Display the DataFrame in Streamlit
    st.dataframe(df)