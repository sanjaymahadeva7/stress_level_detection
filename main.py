import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the saved Logistic Regression model
model = joblib.load('logistic_regression.pkl')

# Function to make predictions
def predict(features):
    return model.predict([features])

# Streamlit app
st.title("Stress Level Prediction")

# Add custom CSS for background image
background_image_url = 'https://c8.alamy.com/comp/2J2J845/tired-business-person-reducing-level-of-stress-cartoon-character-pulling-arrow-measure-of-stress-at-work-flat-vector-illustration-stress-crisis-p-2J2J845.jpg'  # Replace with your image URL
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)



# User inputs for features
st.header("Input Features")
snoring_rate = st.number_input("snoring rate")
respiration_rate = st.number_input("respiration rate")
body_temperature = st.number_input("body temperature")
limb_movement = st.number_input("limb movement")
blood_oxygen = st.number_input("blood oxygen")
eye_movement = st.number_input("eye movement")
sleeping_hours = st.number_input("sleeping hours")
heart_rate = st.number_input("heart rate")

# Predict button
if st.button("Predict Stress Level"):
    features = [snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate]
    prediction = predict(features)
    st.write(f"Predicted Stress Level: {prediction[0]}")

# File uploader for dataset
st.header("Upload Dataset for Batch Prediction")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded dataset
    data = pd.read_csv(uploaded_file)

    # Rename columns to match those used during model training
    data.rename(columns={'sr': 'snoring rate', 'rr':'respiration rate', 't': 'body temperature', 'lm':'limb movement',
                            'bo':'blood oxygen', 'rem':'eye movement', 'sr.1':'sleeping hours', 'hr':'heart rate',
                            'sl':'stress level'}, inplace=True)

    st.write("Uploaded Dataset:")
    st.write(data)

    # Separate features and target
    if 'stress level' in data.columns:
        X = data.drop('stress level', axis=1)
        y = data['stress level']
    else:
        X = data
        y = None

    # Make predictions on the dataset
    predictions = model.predict(X)
    data['Predicted'] = predictions

    if y is not None:
        data['Actual'] = y
        comparison = data[['Actual', 'Predicted']]
        st.write("Comparison of Actual and Predicted Values:")
        st.write(comparison)
        
        # Evaluate the model
        accuracy = accuracy_score(y, predictions)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y, predictions))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y, predictions))
    else:
        st.write("Predictions:")
        st.write(data)

# To run the app, save this script as app.py and run `streamlit run app.py` in your terminal.
