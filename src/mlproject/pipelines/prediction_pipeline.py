import os
import sys
import pickle
import pandas as pd
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_transformation import DataTransformation

class PredictionPipelineConfig:
          model_file_path = os.path.join("artifacts", "model.pkl")
          preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")
          


class PredictionPipeline:
    def __init__(self, model_file_path, preprocessor_file_path):
        self.model_file_path = model_file_path
        self.preprocessor_file_path = preprocessor_file_path
        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()

    def load_model(self):
        try:
            with open(self.model_file_path, "rb") as model_file:
                model = pickle.load(model_file)
            return model
            logging.info("loading the trained model done")
        except Exception as e:
            raise CustomException("Failed to load the trained model.", e)

    def load_preprocessor(self):
        try:
            with open(self.preprocessor_file_path, "rb") as preprocessor_file:
                preprocessor = pickle.load(preprocessor_file)
            return preprocessor
            logging.info("loading the data preprocessor done")
        except Exception as e:
            raise CustomException("Failed to load the data preprocessor.", e)

    def preprocess_data(self, input_data):
        try:
            
            # Perform data transformation using the loaded preprocessor
            transformed_data = self.preprocessor.transform(input_data)
            
            return transformed_data
            logging.info("Transformation on input data done")
        except Exception as e:
            raise CustomException("Failed to preprocess input data.", e)

    def make_prediction(self, input_data):
        try:
            # Perform data preprocessing
            preprocessed_data = self.preprocess_data(input_data)
            
            # Make predictions using the loaded model
            predictions = self.model.predict(preprocessed_data)

            return predictions
        except Exception as e:
            raise CustomException("Failed to make predictions.", e)