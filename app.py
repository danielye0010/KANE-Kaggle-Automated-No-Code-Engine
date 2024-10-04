from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import threading
import pandas as pd
import zipfile
from autogluon.tabular import TabularPredictor
import sweetviz as sv
import uuid
from threading import Lock

app = Flask(__name__)

# Thread-safe dictionary to store progress
progress_data = {}
progress_lock = Lock()

def download_data(competition_name, download_path):
    os.makedirs(download_path, exist_ok=True)
    os.system(f'kaggle competitions download -c {competition_name} -p "{download_path}"')
    for file in os.listdir(download_path):
        if file.endswith('.zip'):
            zip_path = os.path.join(download_path, file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            os.remove(zip_path)

def detect_problem_type(train_data, label_column):
    unique_values = train_data[label_column].nunique()
    dtype = train_data[label_column].dtype
    if pd.api.types.is_numeric_dtype(dtype):
        if unique_values <= 2:
            return 'binary'
        elif unique_values <= 20:
            return 'multiclass'
        else:
            return 'regression'
    else:
        return 'multiclass' if unique_values > 2 else 'binary'

def get_eval_metrics():
    return [
        'auto',
        'accuracy',
        'roc_auc',
        'log_loss',
        'f1',
        'precision',
        'recall',
        'rmse',
        'mse',
        'mae',
        'r2'
    ]

def train_and_predict(train_data, test_data, label_column, problem_type, eval_metric, time_limit, preset):
    if problem_type == 'auto':
        problem_type = detect_problem_type(train_data, label_column)

    predictor = TabularPredictor(label=label_column, problem_type=problem_type, eval_metric=eval_metric)
    predictor.fit(train_data, presets=preset, time_limit=time_limit)
    predictions = predictor.predict(test_data)
    return predictions

def run_training(competition_name, label_column, problem_type, eval_metric, time_limit, id_column, preset, task_id):
    try:
        with progress_lock:
            progress_data[task_id] = 0

        # Download data
        download_path = os.path.join('./kaggle_data', competition_name)
        download_data(competition_name, download_path)
        with progress_lock:
            progress_data[task_id] = 10

        train_file = os.path.join(download_path, 'train.csv')
        test_file = os.path.join(download_path, 'test.csv')

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Could not find 'train.csv' or 'test.csv' in {download_path}.")

        # Read data
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        test_ids = test[id_column]
        with progress_lock:
            progress_data[task_id] = 20

        # Prepare data
        test_data_for_training = test.copy()
        with progress_lock:
            progress_data[task_id] = 30

        # Detect problem type if needed
        if problem_type == 'auto':
            problem_type = detect_problem_type(train, label_column)
        with progress_lock:
            progress_data[task_id] = 40

        # Start training
        predictor = TabularPredictor(label=label_column, problem_type=problem_type, eval_metric=eval_metric)
        predictor.fit(train, presets=preset, time_limit=time_limit)
        with progress_lock:
            progress_data[task_id] = 80

        # Make predictions
        predictions = predictor.predict(test_data_for_training)
        with progress_lock:
            progress_data[task_id] = 90

        # Save submission file
        submission = pd.DataFrame({id_column: test_ids, label_column: predictions})
        submission_file = os.path.join(download_path, f'{competition_name}_submission.csv')
        submission.to_csv(submission_file, index=False)
        with progress_lock:
            progress_data[task_id] = 100

    except Exception as e:
        with progress_lock:
            progress_data[task_id] = 'error'
            progress_data[task_id + '_error'] = str(e)
    finally:
        pass

@app.route('/')
def index():
    eval_metrics = get_eval_metrics()
    time_limit_options = ['5 Minutes', '30 Minutes', '1 Hour', '2 Hours', '10 Hours']
    presets = ['best_quality', 'good_quality', 'medium_quality']
    problem_types = ['auto', 'binary', 'multiclass', 'regression']
    return render_template('index.html', eval_metrics=eval_metrics, time_limit_options=time_limit_options,
                           presets=presets, problem_types=problem_types)

@app.route('/start_training', methods=['POST'])
def start_training():
    # Get form data
    competition_name = request.form.get('competition_name').strip()
    label_column = request.form.get('label_column').strip()
    id_column = request.form.get('id_column').strip() or 'Id'
    problem_type = request.form.get('problem_type')
    eval_metric = request.form.get('eval_metric')
    time_limit_str = request.form.get('time_limit')
    preset = request.form.get('preset')

    if not competition_name or not label_column:
        return "Please provide both the competition name and label column.", 400

    # Convert time_limit_str to seconds
    time_limit_options = {
        '5 Minutes': 300,
        '30 Minutes': 1800,
        '1 Hour': 3600,
        '2 Hours': 7200,
        '10 Hours': 36000
    }
    time_limit = time_limit_options.get(time_limit_str, 3600)

    # Generate a unique task id
    task_id = str(uuid.uuid4())
    # Initialize progress
    with progress_lock:
        progress_data[task_id] = 0

    # Start training in a separate thread
    threading.Thread(target=run_training, args=(
        competition_name,
        label_column,
        problem_type,
        eval_metric if eval_metric != 'auto' else None,
        time_limit,
        id_column,
        preset,
        task_id
    )).start()

    # Redirect to a status page, passing the task_id
    return redirect(url_for('status', task_id=task_id))

@app.route('/status/<task_id>')
def status(task_id):
    return render_template('status.html', task_id=task_id)

@app.route('/progress/<task_id>')
def progress(task_id):
    with progress_lock:
        progress = progress_data.get(task_id, None)
        error = progress_data.get(task_id + '_error', None)
    if progress is None:
        # Task not found
        return jsonify({'status': 'unknown'})
    elif progress == 'error':
        return jsonify({'status': 'error', 'error': error})
    else:
        return jsonify({'status': 'progress', 'progress': progress})

@app.route('/generate_eda_report', methods=['POST'])
def generate_eda_report():
    try:
        competition_name = request.form.get('competition_name').strip()
        download_path = os.path.join('./kaggle_data', competition_name)
        train_file = os.path.join(download_path, 'train.csv')

        if not os.path.exists(train_file):
            return "Training data not found. Please download data first.", 400

        # Read the train dataset
        train_data = pd.read_csv(train_file)

        # Generate Sweetviz report
        report = sv.analyze(train_data)
        report_file = os.path.join(download_path, 'sweetviz_report.html')
        report.show_html(report_file)

        return f"EDA report generated and saved at {report_file}."
    except Exception as e:
        return f"Failed to generate EDA report: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
