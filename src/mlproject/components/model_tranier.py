import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object,evaluate_models 



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "logistic_regression": LogisticRegression(),
                "decision_tree": DecisionTreeClassifier(),
                "random_forest": RandomForestClassifier(),
                "svc": SVC(),
                "k_neighbors": KNeighborsClassifier(),
            }

            params = {
                "logistic_regression": {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'max_iter': [1000]
                },

                "decision_tree": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "random_forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "svc": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                },
                "k_neighbors": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                }
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing datasets")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
        except Exception as e:
            raise CustomException(e, sys)