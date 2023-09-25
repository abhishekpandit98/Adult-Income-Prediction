from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig,DataTransformation
from src.mlproject.components.model_tranier import ModelTrainerConfig,ModelTrainer
from src.mlproject.pipelines.prediction_pipeline import PredictionPipelineConfig,PredictionPipeline
import sys
import streamlit as st
import pandas as pd

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        # data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        # data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr,preprocessor_file_path=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        # Model Training
        # model_trainer=ModelTrainer()
        # print(model_trainer.initiate_model_trainer(train_arr,test_arr))

        # Initialize Streamlit UI
        logging.info("The Streamlit execution has started")

        st.title("Machine Learning Prediction App")
        st.write("Enter the data for prediction:")

        # Create input fields for user input
        age = st.number_input("Age", min_value=0, max_value=100)
        workclass = st.selectbox("Workclass", [' State-gov', ' Self-emp-not-inc', ' Private', ' Federal-gov',
                                       ' Local-gov', ' Self-emp-inc', ' Without-pay',
                                       ' Never-worked'])
        education = st.selectbox("Education", [' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',
                                       ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th',
                                       ' Doctorate', ' Prof-school', ' 5th-6th', ' 10th', ' 1st-4th',
                                       ' Preschool', ' 12th'])
        education_num = st.number_input("Education Number", min_value=0, max_value=20)
        marital_status = st.selectbox("Marital Status", [' Never-married', ' Married-civ-spouse', ' Divorced',
                                                 ' Married-spouse-absent', ' Separated', ' Married-AF-spouse',
                                                 ' Widowed'])
        occupation = st.selectbox("Occupation", [' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners',
                                         ' Prof-specialty', ' Other-service', ' Sales', ' Craft-repair',
                                         ' Transport-moving', ' Farming-fishing', ' Machine-op-inspct',
                                         ' Tech-support', ' Protective-serv', ' Armed-Forces',
                                         ' Priv-house-serv'])
        race = st.selectbox("Race", [' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other'])
        sex = st.selectbox("Sex", [' Male', ' Female'])
        hours_per_week = st.number_input("Hours Per Week", min_value=0, max_value=168)
        country = st.selectbox("Country", [' United-States', ' Cuba', ' Jamaica', ' India', ' Mexico',
                                    ' South', ' Puerto-Rico', ' Honduras', ' England', ' Canada',
                                    ' Germany', ' Iran', ' Philippines', ' Italy', ' Poland',
                                    ' Columbia', ' Cambodia', ' Thailand', ' Ecuador', ' Laos',
                                    ' Taiwan', ' Haiti', ' Portugal', ' Dominican-Republic',
                                    ' El-Salvador', ' France', ' Guatemala', ' China', ' Japan',
                                    ' Yugoslavia', ' Peru', ' Outlying-US(Guam-USVI-etc)', ' Scotland',
                                    ' Trinadad&Tobago', ' Greece', ' Nicaragua', ' Vietnam', ' Hong',
                                    ' Ireland', ' Hungary', ' Holand-Netherlands'])
        logging.info("The Streamlit input field creation done")

       

        # Create a button to trigger the prediction
        if st.button("Predict"):
            # Create a DataFrame from user input
            input_data = pd.DataFrame({
                'age': [age],
                 'workclass': [workclass],
                'education': [education],
                'education-num': [education_num],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'race': [race],
                'sex': [sex],
                'hours-per-week': [hours_per_week],
                'country': [country]
            })
            logging.info("The user input dataframe created")

            # Make predictions using the prediction pipeline
            model_file_path = PredictionPipelineConfig.model_file_path
            preprocessor_file_path = PredictionPipelineConfig.preprocessor_file_path
            logging.info("Loading model_file_path and preprocessor_file_path done")
            prediction_pipeline = PredictionPipeline(model_file_path, preprocessor_file_path)
            logging.info("predication is going to start on user input")
            predictions = prediction_pipeline.make_prediction(input_data)

            # Display the predictions
            st.write("The predicated salary of the person is:")
            st.write(predictions)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
