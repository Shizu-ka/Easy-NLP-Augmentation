import pandas as pd

def check_text_input(text):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if text.strip() == "":
        raise ValueError("Input text cannot be empty.")

def check_df_input(df, text_column):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame.")

def check_classes_to_augment(classes_to_augment):
    if not isinstance(classes_to_augment, list) or not all(isinstance(i, (str)) for i in classes_to_augment):
        raise ValueError("classes_to_augment must be a list of strings.")

def check_augmentation_percentage(augmentation_percentage):
    if not (0 <= augmentation_percentage <= 1):
        raise ValueError("augmentation_percentage must be between 0 and 1.")

def check_random_state(random_state):
    if not isinstance(random_state, int):
        raise TypeError("random_state must be an integer.")

def check_weights(weights, expected_length):
    if not isinstance(weights, list) or not all(isinstance(w, (int, float)) for w in weights) or len(weights) != expected_length:
        raise ValueError(f"weights must be a list of {expected_length} numeric values.")

def handle_row_errors(row, text_column):
    if text_column not in row or not isinstance(row[text_column], str):
        print(f"Invalid text data for row: {row}.")
        return False
    return True
