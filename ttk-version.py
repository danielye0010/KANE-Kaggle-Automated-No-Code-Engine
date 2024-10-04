import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import zipfile
from autogluon.tabular import TabularPredictor


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

    # Train the model using the selected preset
    predictor = TabularPredictor(label=label_column, problem_type=problem_type, eval_metric=eval_metric)
    predictor.fit(train_data, presets=preset, time_limit=time_limit)
    predictions = predictor.predict(test_data)
    return predictions


def run_training(competition_name, label_column, problem_type, eval_metric, time_limit, id_column, progress_var, preset):
    try:
        download_path = os.path.join('./kaggle_data', competition_name)
        download_data(competition_name, download_path)

        train_file = os.path.join(download_path, 'train.csv')
        test_file = os.path.join(download_path, 'test.csv')

        # Check if the files exist
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Could not find 'train.csv' or 'test.csv' in {download_path}.")

        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        test_ids = test[id_column]

        # Ensure PassengerId is not dropped for either train or test
        test_data_for_training = test.copy()

        # Start training
        start_time = time.time()

        # Function to update progress bar
        def update_progress():
            while True:
                elapsed_time = time.time() - start_time
                progress = min(int((elapsed_time / time_limit) * 100), 100)
                progress_var.set(progress)
                if progress >= 100:
                    break
                time.sleep(1)

        # Start progress bar update thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.start()

        # Train model and make predictions
        predictions = train_and_predict(train, test_data_for_training, label_column, problem_type, eval_metric,
                                        time_limit, preset)
        progress_var.set(100)  # Ensure progress bar is full upon completion

        # Save submission file
        submission = pd.DataFrame({id_column: test_ids, label_column: predictions})
        submission_file = os.path.join(download_path, f'{competition_name}_submission.csv')
        submission.to_csv(submission_file, index=False)
        messagebox.showinfo("Success",
                            f"Training completed and submission file saved as {competition_name}_submission.csv.")
    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        # Re-enable the Start button
        start_button.config(state='normal')


def start_training():
    competition_name = competition_name_entry.get().strip()
    label_column = label_column_entry.get().strip()
    problem_type = problem_type_var.get()
    eval_metric = eval_metric_var.get()
    time_limit_str = time_limit_var.get()
    preset = preset_var.get()  # Get the selected preset
    id_column = id_column_entry.get().strip() or 'Id'

    if not competition_name or not label_column:
        messagebox.showwarning("Input Error", "Please provide both the competition name and label column.")
        return

    time_limit_options = {
        '5 Minutes': 300,
        '30 Minutes': 1800,
        '1 Hour': 3600,
        '2 Hours': 7200,
        '10 Hours': 36000
    }
    time_limit = time_limit_options.get(time_limit_str, 3600)

    # Disable Start button to prevent multiple clicks
    start_button.config(state='disabled')

    threading.Thread(target=run_training, args=(
        competition_name,
        label_column,
        problem_type,
        eval_metric if eval_metric != 'auto' else None,
        time_limit,
        id_column,
        progress_var,
        preset  # Pass the selected preset to run_training
    )).start()


# Create the GUI
root = tk.Tk()
root.title("AutoML for Tabular Data")

# Main Frame
main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky='nsew')

# Title Label
title_label = ttk.Label(main_frame, text="AutoML for Tabular Data", font=('Helvetica', 16, 'bold'))
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

# Competition Name
ttk.Label(main_frame, text="Competition Name:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
competition_name_entry = ttk.Entry(main_frame, width=30)
competition_name_entry.grid(row=1, column=1, pady=5)

# Label Column
ttk.Label(main_frame, text="Label Column:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
label_column_entry = ttk.Entry(main_frame, width=30)
label_column_entry.grid(row=2, column=1, pady=5)

# ID Column
ttk.Label(main_frame, text="ID Column:").grid(row=3, column=0, sticky='e', padx=5, pady=5)
id_column_entry = ttk.Entry(main_frame, width=30)
id_column_entry.grid(row=3, column=1, pady=5)

# Problem Type
ttk.Label(main_frame, text="Problem Type:").grid(row=4, column=0, sticky='e', padx=5, pady=5)
problem_type_var = tk.StringVar()
problem_type_var.set('auto')
problem_type_menu = ttk.Combobox(main_frame, textvariable=problem_type_var, state='readonly', width=28)
problem_type_menu['values'] = ('auto', 'binary', 'multiclass', 'regression')
problem_type_menu.grid(row=4, column=1, pady=5)

# Evaluation Metric
ttk.Label(main_frame, text="Evaluation Metric:").grid(row=5, column=0, sticky='e', padx=5, pady=5)
eval_metric_var = tk.StringVar()
eval_metric_var.set('auto')
eval_metric_menu = ttk.Combobox(main_frame, textvariable=eval_metric_var, state='readonly', width=28)
eval_metric_menu['values'] = get_eval_metrics()
eval_metric_menu.grid(row=5, column=1, pady=5)

# Preset
ttk.Label(main_frame, text="Preset:").grid(row=6, column=0, sticky='e', padx=5, pady=5)
preset_var = tk.StringVar()
preset_menu = ttk.Combobox(main_frame, textvariable=preset_var, state='readonly', width=28)
preset_menu['values'] = ('best_quality', 'good_quality', 'medium_quality')  # Preset options
preset_menu.grid(row=6, column=1, pady=5)

# Time Limit
ttk.Label(main_frame, text="Time Limit:").grid(row=7, column=0, sticky='e', padx=5, pady=5)
time_limit_var = tk.StringVar()
time_limit_menu = ttk.Combobox(main_frame, textvariable=time_limit_var, state='readonly', width=28)
time_limit_menu['values'] = ('5 Minutes', '30 Minutes', '1 Hour', '2 Hours', '10 Hours')
time_limit_menu.grid(row=7, column=1, pady=5)

# Start Button
start_button = ttk.Button(main_frame, text="Start Training", command=start_training)
start_button.grid(row=8, column=0, columnspan=2, pady=15)

# Progress Bar
progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(main_frame, variable=progress_var, maximum=100)
progress_bar.grid(row=9, column=0, columnspan=2, sticky='we', padx=5, pady=5)

# Configure grid weights for responsiveness
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=1)

root.mainloop()
