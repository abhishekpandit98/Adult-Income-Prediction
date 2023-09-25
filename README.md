# ADULT_CENSUS_INCOME# Adult_Census_Income_Prediction
Adult Census Income Prediction ineuron internship project

### Software and account Requirement.

1. [Github Account](https://github.com)
2. [VS Code IDE](https://code.visualstudio.com/download)
3. [GIT cli](https://git-scm.com/downloads)
4. [GIT Documentation](https://git-scm.com/docs/gittutorial)


Creating conda environment
```
conda create --prefix ./env python==3.8 -y
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
Here we have define cycle of machine learning model-

1.Data_ingestion.py: In this phase we will get our data from the MySql database.



2.Data_transformation.py: In this phase we will transform our data according to our EDA which we have done in jupyter notebook like all preprocessing steps like scalling our data encoding our target column and also convert cat column to numerical with the help of Onehot encoder.



3.Model_trainer.py: In this phase we will train our model on random forest algorithm and save our model in artifact folder
