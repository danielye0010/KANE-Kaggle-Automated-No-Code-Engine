# AutoML for a 3-Year-Old: Beat 13000+ Data Scientists with 10 Lines of Code in 5 minutes

## Short Description
A no-code AutoML tool with a simple GUI—train a model in 10 minutes, beat 13,000+ Kaggle competitors with just 10 lines of code!

---

## Overview

This tool makes machine learning **so simple** that even a 3-year-old could do it! With a beginner-friendly interface, you can train powerful models on Kaggle competitions like Titanic without writing any code. Just input a few details, and the tool will download data, train the model, and generate a ready-to-submit file.

---

## Key Features

 **No Coding Needed**: Simply use the GUI to select options—AutoGluon handles the rest.
- **Fast & Powerful**: Train models in as little as 5 minutes and get results that can compete with thousands of data scientists.
- **Real-Time Feedback**: Progress bar keeps you updated as your model trains.
- **Auto Submission File**: Automatically generates a `submission.csv` file ready for Kaggle.
- **Multiple Evaluation Metrics**: Choose from metrics like accuracy, AUC, F1, RMSE, and more to match your competition’s requirements.
- **Automatic Dataset Handling**: Automatically downloads and unzips Kaggle datasets—no manual steps required.
- **Preset Flexibility**: Easily choose between `good_quality` for fast training or `best_quality` for maximum performance.
- **Dynamic Problem Type Detection**: The tool automatically detects whether your problem is binary classification, multiclass classification, or regression—no need to manually configure.
- **Time Limit Options**: Set time limits from 5 minutes to 10 hours for optimal training duration based on your resources and needs.
- **User-Friendly Customization**: Input fields for competition name, label column, and ID column make it easy to adapt to any Kaggle competition.
- **Versatile Problem Solving**: Whether you’re tackling a simple binary classification like Titanic or a complex regression task, this tool adapts to various types of machine learning challenges.

---

## Example: Beat 13,000 Data Scientists on Titanic in 10 Minutes

The **Titanic - Machine Learning from Disaster** competition on Kaggle is a classic binary classification problem. The goal is to predict whether a passenger survived or not based on various features such as their age, ticket class, gender, and more. With over 13,000 participants, it's a highly competitive challenge.

Using this tool, you can train a machine learning model on the Titanic dataset in just **10 minutes** and achieve a **top 4% ranking**, beating thousands of data scientists—all with just 10 lines of code and no prior machine learning experience.

Here’s the code to get started:

```python
run_kaggle_automl(
    competition_name='titanic',
    label_column='Survived',
    id_column='PassengerId',
    time_limit=600,  # 10 minutes
    presets='best_quality'
)

