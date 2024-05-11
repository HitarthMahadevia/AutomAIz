
from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("logo2.jpg", width=150)
    st.title("AutomAIz")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This application allows you to build an automated Machine Learning pipeline using StreamLit, Pandas Profiling, and PyCaret. It's magic!")


if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

# Model building
if choice == "Modelling":
    st.write("Note:- At present we can work only on Regression problems. We will be adding more models soon. Stay tunned!")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        
        # Regression modeling (example: Linear Regression)
        best_model = compare_models(include=['lr'])  # Include other regression models as needed
        compare_df = pull()
        st.dataframe(compare_df)
        
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")