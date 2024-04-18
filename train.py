from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd

# Loading data
df = pd.read_csv('queries1.csv')

X = df['String']
y = df['Class']

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that first transforms your data using TF-IDF and then fits it to a SVC
model = make_pipeline(TfidfVectorizer(), SVC(probability=True))

# Train your model using the training data
model.fit(X_train, y_train)

# Evaluate your model using the testing data
print("Model Accuracy: ", model.score(X_test, y_test))

# Your custom string
custom_string = "different cheese"

# Transform your string into a TF-IDF vector
custom_string_transformed = model.named_steps['tfidfvectorizer'].transform([custom_string])

# Make prediction
prediction = model.named_steps['svc'].predict(custom_string_transformed)

print(prediction)



from joblib import dump

# Save the model to a file
dump(model, 'model.joblib')


# from joblib import load

# # Load the model from a file
# model_2 = load('model.joblib')