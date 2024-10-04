import os
import pandas as pd
from autogluon.tabular import TabularPredictor


def run_kaggle_automl(competition, label, id_col, time):
    path = f'./kaggle_data/{competition}'
    if not os.path.exists(f'{path}/train.csv') or not os.path.exists(f'{path}/test.csv'):
        os.system(f'kaggle competitions download -c {competition} -p {path} && unzip -o {path}/*.zip -d {path}')
    train, test = pd.read_csv(f'{path}/train.csv'), pd.read_csv(f'{path}/test.csv')
    predictions = TabularPredictor(label=label).fit(train.drop(columns=[id_col]), time_limit=time).predict(
        test.drop(columns=[label], errors='ignore'))
    pd.DataFrame({id_col: test[id_col], label: predictions}).to_csv(f'{path}/submission.csv', index=False)


run_kaggle_automl('titanic', 'Survived', 'PassengerId', 300)  # replace with actual value
