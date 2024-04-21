import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Assuming the CSV file is in the same directory
df = pd.read_csv('recipesFinal.csv', names=['Recipe', 'Ingredients', 'Instructions'])

# Preprocess the ingredients
df['Ingredients'] = df['Ingredients'].str.replace(',', ' ')

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the vectorizer on our corpus
tfidf_matrix = vectorizer.fit_transform(df['Ingredients'])

def get_top_recipes(query, top_n=5):
    # Transform the query using the same vectorizer
    query_vec = vectorizer.transform([query])
    
    # Compute the cosine similarity between the query and all recipes
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get the top_n most similar recipes
    most_similar_recipe_indices = cosine_similarities.argsort()[:-top_n - 1:-1]
    
    # Get the top_n most similar recipes' probabilities
    most_similar_recipe_probs = cosine_similarities[most_similar_recipe_indices]
    
    # Create a DataFrame with the recipe names and probabilities
    top_recipes = pd.DataFrame({
        'Recipe': df['Recipe'].iloc[most_similar_recipe_indices],
        'Cosine Score': most_similar_recipe_probs
    })
    
    return top_recipes