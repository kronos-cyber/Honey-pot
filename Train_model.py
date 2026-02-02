import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

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
    "label": [1,1,1,1,0,0,0]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

model = LogisticRegression()
model.fit(X, df["label"])

pickle.dump(model, open("scam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained & saved")
