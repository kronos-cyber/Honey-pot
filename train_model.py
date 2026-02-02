import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Dataset
data = {
    "text": [
        "Your bank account is blocked click link",
        "Urgent KYC update required",
        "Win lottery now",
        "OTP shared please verify",
        "Hello how are you",
        "Meeting at 5 pm",
        "Order shipped successfully"
    ],
    "label": [1, 1, 1, 1, 0, 0, 0]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Initialize the TfidfVectorizer and transform text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# Train the model
model = LogisticRegression()
model.fit(X, df["label"])

# Save the logistic regression model
with open("scam_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
print("Model saved as scam_model.pkl")

# Save the vectorizer
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
print("Vectorizer saved as vectorizer.pkl")