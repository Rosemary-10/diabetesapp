import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pickle

# Load model
loaded_model = pickle.load(open('diabetes_classifier', 'rb'))

# Load dataset
load = pd.read_csv('diabetes.csv')

# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return (
        'Unfortunately, you are susceptible to diabetes.'
        if prediction[0] == 0
        else 'Congratulations! You are not susceptible to diabetes.'
    )

# Dashboard page
def dashboard_page():
    st.title('Diabetes Prediction Dashboard')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Pregnancies', value=0)
        Glucose = st.number_input('Glucose', value=0)
        BloodPressure = st.number_input('Blood Pressure', value=0)
        SkinThickness = st.number_input('Skin Thickness', value=0)

    with col2:
        Insulin = st.number_input('Insulin', value=0)
        BMI = st.number_input('BMI', value=0.0)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', value=0.0)
        Age = st.number_input('Age', value=0)

    if st.button('Predict Diabetes Risk'):
        try:
            input_data = [
                int(Pregnancies),
                int(Glucose),
                int(BloodPressure),
                int(SkinThickness),
                int(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                int(Age)
            ]
            result = diabetes_prediction(input_data)
            st.success(result)

            # Optional: show extra insights
            st.subheader('Insights')
            st.markdown('Men are more susceptible to diabetes than women.')
        except ValueError:
            st.error('Enter valid input values')

# Chart page
def chart_page():
    st.title('Diabetes Data Visualization')
    fig = px.histogram(load, x='Age', color='Outcome', barmode='group')
    st.plotly_chart(fig)

# Main app
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Select Page', ['Chart', 'Form Inputs'])

    if page == 'Chart':
        chart_page()
    elif page == 'Form Inputs':
        dashboard_page()

if __name__ == '__main__':
    main()
