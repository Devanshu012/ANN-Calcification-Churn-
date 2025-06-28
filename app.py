import streamlit as st 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Import the trained model
model = tf.keras.models.load_model('model.keras')

### Load the encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)
    
## Streamlit App
st.title('Customer Churn Prediction')

#  User Input
credit_score = st.number_input('Credit Score')
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])
estimated_salary = st.number_input('Estimated Salary')

# Prepare the Input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## Encode Categorical Variables
input_data['Gender']=label_encoder_gender.transform(input_data['Gender'])

# One Hot Encoded Geography
geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)

## Scaling the input data
input_scaled=scaler.transform(input_data)

##  Predict Churn
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]
st.write(f"Churn Probability:  {prediction_proba:.4f}")
if prediction_proba>0.5:
    st.write("The customer is likely to curn.")
else:
    st.write("The customer is not likely to curn.")