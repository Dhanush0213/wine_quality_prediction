# wine_quality_prediction
Wine Quality Prediction Using Linear Regression Machine Learning
This repository contains the code and resources for predicting the quality of wines using linear regression machine learning algorithms. The project focuses on analyzing various features of wines and training a linear regression model to predict their quality based on those features.

Dataset
The dataset used for this project is the Wine Quality Dataset from the UCI Machine Learning Repository. It consists of red and white wine samples, each with 11 physicochemical features such as acidity, pH, alcohol content, etc., along with a quality rating ranging from 0 to 10. The dataset is available in the data directory.

Dependencies
The following dependencies are required to run the code in this repository:

Python 3.x
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
You can install the required dependencies using the following command:

pip install -r requirements.txt
Code Structure
data/: Directory containing the dataset files.
notebooks/: Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
src/: Python scripts for preprocessing, model training, and evaluation.
models/: Directory to save trained models.
utils/: Utility functions and classes used in the project.
README.md: This file, providing an overview of the repository.
Usage
To train and evaluate the linear regression model, follow these steps:

Clone the repository:

git clone https://github.com/your-username/wine-quality-prediction.git
cd wine-quality-prediction
Install the dependencies:

pip install -r requirements.txt
Explore the Jupyter notebooks in the notebooks/ directory to understand the project workflow and data analysis.

Preprocess the data:

python src/data_preprocessing.py
This script performs data cleaning, feature engineering, and splitting the dataset into training and testing sets.

Train the linear regression model:

python src/train_model.py
This script trains a linear regression model on the preprocessed data and saves it in the models/ directory.

Evaluate the model:

python src/evaluate_model.py
This script evaluates the trained model using various metrics and generates visualizations.

Results
The results of the linear regression model, including evaluation metrics and visualizations, can be found in the results/ directory.

Contributing
Contributions to this repository are always welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code for personal and commercial purposes.
