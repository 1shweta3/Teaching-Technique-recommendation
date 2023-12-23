import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


data = pd.read_csv('data.csv')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


data['Technique_Code'] = pd.Categorical(data['Technique']).codes

# Split the dataset into features (X) and target labels (y)
X = data[['Age', 'Intellectual_Ability', 'Topic']]
y = data['Technique']

X = pd.get_dummies(X, columns=['Age', 'Intellectual_Ability', 'Topic'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
clf = MultinomialNB()
clf.fit(X_train, y_train)

X_test_counts =vectorizer.fit_transform(X_test)
y_pred = clf.predict(X_test_counts)



# Streamlit app
st.header("Technique Prediction App")

# User input
age = st.selectbox("Select Age:", ['Young', 'Middle', 'Old'])
intellectual_ability = st.selectbox("Select Intellectual Ability:", ['Low', 'Medium', 'High'])
topic = st.selectbox("Select topic",data['Topic'])

# Predict button
if st.button("Predict Technique"):
    # Prepare user input data
    user_input = pd.DataFrame({
        'Age': [age],
        'Intellectual_Ability': [intellectual_ability],
        'Topic': [topic]
    })

    # Preprocess user input
    user_input = pd.get_dummies(user_input, columns=['Age', 'Intellectual_Ability', 'Topic'])
    
    # Ensure that the user input data contains all training feature names
    for feature_name in X_train.columns:
        if feature_name not in user_input.columns:
            user_input[feature_name] = 0

    # Now, the user_input_data should have all the necessary feature columns
    user_input = user_input[X_train.columns]

    # Make a prediction
    user_input_counts = vectorizer.transform(user_input)
    prediction = clf.predict(user_input_counts)

    # Display the prediction
    st.success(f"The predicted technique for the provided input is: {prediction[0],prediction[1],prediction[2]}")

