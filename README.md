
Adult Census Income Prediction ineuron internship project
<br>

Application URL Links : [InsurancePremiumPredictor](https://abhishekpandit98-adult-income-prediction-app-uy6n8o.streamlit.app/)

<br>

# Machine Learning Income Prediction App

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Key Features](#key-features)
4. [Getting Started](#getting-started)
5. [Dependencies](#dependencies)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Folder Structure](#folder-structure)


## 1. Introduction

The Machine Learning Income Prediction App is a web-based application that predicts a person's income based on various input features, such as age, education, and hours worked per week. This README provides an overview of the project, instructions for setting it up, and details on how to use the app.

## 2. Project Overview

The project's main goal is to create an easy-to-use income prediction tool that leverages machine learning models. The app takes user input, preprocesses it, and provides income predictions based on a trained model.

## 3. Key Features

- User-friendly web interface for input and predictions.
- Integration of machine learning models for accurate income predictions.
- Support for various input features and demographics.
- Clear and intuitive presentation of results.

## 4. Getting Started

To get started with the Machine Learning Income Prediction App, follow the installation and usage instructions below.

## 5. Dependencies

The following dependencies are required to run the app:

- Python
- pandas
- scikit-learn
- Streamlit
- ### Software and account Requirement.

1. [Github Account](https://github.com)
2. [VS Code IDE](https://code.visualstudio.com/download)
3. [GIT cli](https://git-scm.com/downloads)
4. [GIT Documentation](https://git-scm.com/docs/gittutorial)
  

## 6. Installation



Creating conda environment
```
conda create --prefix ./env python==3.9 -y
```
```
conda activate venv/
```
OR 
```
conda activate ./env
```

```
pip install -r requirements.txt
```

To Add files to git
```
git add .
```

OR
```
git add <file_name>
```

> Note: To ignore file or folder from git we can write name of file/folder in .gitignore file

To check the git status 
```
git status
```
To check all version maintained by git
```
git log
```

To create version/commit all changes by git
```
git commit -m "message"
```

To send version/changes to github
```
git push origin main
```

To check remote url 
```
git remote -v
```
## 7. Usage
Run the Streamlit app:
```
streamlit run app.py
```
Input the required data, such as age, education, and hours worked per week.

Click the "Predict" button to see the predicted income.

## 8. Folder Structure

- `data_ingestion.py`: Handles data loading and preprocessing.
- `data_transformation.py`: Manages data transformation and preprocessing.
- `model_trainer.py`: Handles model training and evaluation.
- `prediction_pipeline.py`: Contains the PredictionPipeline class for making predictions.
- `app.py`: Main application script for UI and integration.
- `artifacts/`: Folder for storing model and preprocessor objects.
