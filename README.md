# AutoML solution with GUI -  KANE: Kaggle AutoML No-Code Engine

## Short Description

KANE (Kaggle AutoML No-Code Engine) is a web-based application designed to simplify and automate participation in Kaggle competitions. Built on top of the AutoGluon framework, KANE offers an intuitive, no-code interface to configure, execute, and optimize machine learning workflows with minimal effort. Its features include parameter configuration, automated backend processes, advanced training strategies, and real-time monitoring.

---

## Features

### 1. Intuitive Parameter Configuration via Dropdown Menus
KANE provides a user-friendly interface for configuring machine learning tasks without any coding. Users can customize the following options:
- **Competition Name**: Specify the Kaggle competition to automatically download and prepare datasets.
- **Label Column**: Define the target variable for prediction.
- **Problem Type**: Choose between regression, binary classification, or multiclass classification—or let KANE auto-detect the problem type.
- **Evaluation Metric**: Select metrics like accuracy or RMSE to guide optimization.
- **Time Limit**: Set predefined training durations (e.g., 5 minutes, 1 hour) to balance performance and resource usage.
- **Training Presets**: Adjust model depth and computational intensity with presets like `'best_quality'`, `'good_quality'`, or `'medium_quality'`.

---

### 2. Automated Backend Processes
KANE automates critical backend workflows to ensure seamless execution:
- **Automated Data Acquisition**: Downloads and prepares competition data based on user inputs, handling extraction and organization.
- **No-Code Model Training**: Initiates model training with AutoGluon managing preprocessing, model selection, and optimization.
- **Submission File Generation**: Automatically formats and creates Kaggle-compatible submission files.
- **Exploratory Data Analysis (EDA)**: Generates detailed reports using the Sweetviz framework, offering visualizations and statistical insights.
- **Progress Monitoring and Error Handling**: Includes a real-time progress bar and mechanisms to detect and report errors, providing actionable feedback.

---

### 3. Advanced Training Strategies
KANE leverages AutoGluon’s state-of-the-art training strategies to optimize model performance:
- **Data Preprocessing**: Handles diverse data types (numerical, categorical, text, date/time) with model-agnostic transformations.
- **Multi-Model Ensembling**: Combines models like boosted trees, neural networks, and random forests using multi-layer stacking for robust predictions.
- **Repeated k-Fold Bagging**: Trains multiple versions of models on different data partitions, reducing overfitting and ensuring stability.
- **Hyperparameter Optimization**: Uses Ray Tune to explore hyperparameter spaces with efficient strategies like random search and Bayesian optimization.
- **Resource and Time Management**: Allocates training time adaptively, prioritizing models with the highest potential impact on accuracy.

---

### 4. Real-Time Monitoring and Feedback
KANE offers transparency and reliability through:
- **Real-Time Progress Tracking**: A progress bar displays the status of tasks such as data preparation, training, and submission generation.
- **Error Detection and Reporting**: Alerts users to issues, ensuring quick resolution and uninterrupted workflows.

---

## Getting Started

Follow the steps below to set up and launch KANE on your local machine.

### Clone the Repository
Clone the KANE repository from GitHub to your local machine:
```bash
git clone https://github.com/danielye0010/KANE-Kaggle-Automated-No-Code-Engine.git
```
### Launch the Application
```bash
pip install -r requirements.txt
python app.py
```


## MIT License
KANE is licensed under the MIT License, allowing free use, modification, and distribution with attribution. The software is provided "as is," without warranty of any kind. For full details, see the LICENSE file.
