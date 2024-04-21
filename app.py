import streamlit as st
import pandas as pd
import os
import random
from PIL import Image
from train import get_top_recipes

df1 = pd.read_csv('recipesFinal.csv')
class_list  = os.listdir('101_food_classes_10_percent/test/')
csv_list = sorted(list(df1[df1.columns[0]]))
dish_to_class = dict(zip(csv_list,sorted(class_list)))

# Load the data
df = pd.read_csv('recipesFinal.csv')

st.title("FoodieFinder: A Comprehensive Culinary Keyword Dataset")
st.write("Enter a keyword and we'll predict the top 5 dishes related to it!")

# Create a text input for the user
user_input = st.text_input("Enter your ingredients here")

# When the user enters text, make a prediction
if user_input:
    # Get top recipes
    st.markdown("### Top dishes:")
    top_recipes = get_top_recipes(user_input)
    st.write(top_recipes)


    # Display five images for each top recipe
    for dish in top_recipes['Recipe']:
        dish1 = dish_to_class[dish]
        image_dir = os.path.join('test', dish1)
        images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]
        if images:
            selected_images = random.sample(images, min(5, len(images)))
            cols = st.columns(len(selected_images))
            for i, image_path in enumerate(selected_images):
                image = Image.open(image_path)
                cols[i].image(image, caption=dish, use_column_width=True)