import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import shap # Used for XAI
# Load trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('encoder_gender.pkl', 'rb') as file:
    encoder_gender = pickle.load(file)

with open('encoder_geo.pkl', 'rb') as file:
    encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# SHAP background data, used for explanation training and loading in cache
@st.cache_resource
def load_explainer(_model, _scaler):
    background_data = np.zeros((50, len(_scaler.feature_names_in_)))
    return shap.DeepExplainer(_model, background_data)

explainer = load_explainer(model, scaler)

# Streamlit app
st.title('Customer Churn Prediction')

# Inputs
geography = st.selectbox('Geography', encoder_geo.categories_[0])
gender = st.selectbox('Gender', encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Crredit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0 , 10)
num_of_products = st.slider('Number of Produccts', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Preparing input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[gender],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

# Encoding Gender column
input_data['Gender'] = encoder_gender.transform(input_data['Gender'])

# Encoding Geography column
geography_encoded = encoder_geo.transform(pd.DataFrame({'Geography': [geography]})).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=encoder_geo.get_feature_names_out(['Geography']))

# Concating Encoded Geography column
input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)
input_data = input_data[scaler.feature_names_in_]

# Scaling Input Data
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_probab = prediction[0][0]

# --------- Explanation -------------

#  SHAP values 
shap_values = explainer.shap_values(input_data_scaled)

shap_df = pd.DataFrame({
    'Feature' : scaler.feature_names_in_ , 
    'Impact' : shap_values[0].flatten()
})

shap_df = shap_df.sort_values(by = 'Impact', key=abs , ascending= False)

st.write(f'Chrun Probability:{prediction_probab:.3f}')

if prediction_probab > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

st.subheader("Reasonn for Prediction")

top_feature = shap_df.head(5)

for i in range(len(top_feature)):
    feature = top_feature.iloc[i]['Feature']
    impact = top_feature.iloc[i]['Impact']

    if impact > 0:
        st.write(f"{feature} increases churn risk by {impact:.2f}")
    else:
        st.write(f" {feature} decreases churn risk by {abs(impact):.2f}")

st.subheader("Feature Impact")
st.bar_chart(top_feature.set_index('Feature'))
