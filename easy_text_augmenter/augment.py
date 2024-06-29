import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac 
import random
from easy_text_augmenter import error_handling as eh

def translate(text, aug):
    try:
        eh.check_text_input(text)
        translated = aug.augment(text)
        if not translated or not isinstance(translated, list) or not translated[0]:
            raise ValueError("Augmentation failed to produce valid output")
        return translated[0]
    except Exception as e:
        print(f"Error during translation: {e}")
        return text

def augment_random_word(df, classes_to_augment, augmentation_percentage, text_column, random_state=42, weights=[0.5, 0.3, 0.2]):
    try:
        eh.check_df_input(df, text_column)
        eh.check_classes_to_augment(classes_to_augment)
        eh.check_augmentation_percentage(augmentation_percentage)
        eh.check_random_state(random_state)
        eh.check_weights(weights, 3)

        aug_swap = naw.RandomWordAug(action="swap")
        aug_del = naw.RandomWordAug(action="delete")
        aug_split = naw.SplitAug()

        augmented_rows = []

        for label in classes_to_augment:
            class_df = df[df['label'] == label]
            if class_df.empty:
                print(f"No data found for label '{label}'")
                continue
            num_samples_to_augment = int(len(class_df) * augmentation_percentage)
            sampled_df = class_df.sample(num_samples_to_augment, random_state=random_state)

            for _, row in sampled_df.iterrows():
                if not eh.handle_row_errors(row, text_column):
                    continue
                aug_choice = random.choices(
                    [aug_swap, aug_del, aug_split],
                    weights=weights,
                    k=1
                )[0]
                translated_text = translate(row[text_column], aug=aug_choice)
                augmented_rows.append({text_column: translated_text, 'label': row['label']})

        if not augmented_rows:
            raise ValueError("No valid augmented rows were created")
        
        augmented_df = pd.DataFrame(augmented_rows)
        df = pd.concat([df, augmented_df], ignore_index=True)

        return df

    except Exception as e:
        print(f"Error during augmentation: {e}")
        return df

def augment_random_character(df, classes_to_augment, augmentation_percentage, text_column, random_state=42, weights=[0.2, 0.2, 0.2, 0.2, 0.2]):
    try:
        eh.check_df_input(df, text_column)
        eh.check_classes_to_augment(classes_to_augment)
        eh.check_augmentation_percentage(augmentation_percentage)
        eh.check_random_state(random_state)
        eh.check_weights(weights, 5)
        
        aug_ocr = nac.OcrAug()
        aug_keyboard = nac.KeyboardAug()
        aug_insert = nac.RandomCharAug(action="insert")
        aug_swap = nac.RandomCharAug(action="swap")
        aug_delete = nac.RandomCharAug(action="delete")

        augmented_rows = []

        for label in classes_to_augment:
            class_df = df[df['label'] == label]
            if class_df.empty:
                print(f"No data found for label '{label}'")
                continue
            num_samples_to_augment = int(len(class_df) * augmentation_percentage)
            sampled_df = class_df.sample(num_samples_to_augment, random_state=random_state)

            for _, row in sampled_df.iterrows():
                if not eh.handle_row_errors(row, text_column):
                    continue
                aug_choice = random.choices(
                    [aug_ocr, aug_keyboard, aug_insert, aug_swap, aug_delete],
                    weights=weights,
                    k=1
                )[0]
                translated_text = translate(row[text_column], aug=aug_choice)
                augmented_rows.append({text_column: translated_text, 'label': row['label']})

        augmented_df = pd.DataFrame(augmented_rows)
        df = pd.concat([df, augmented_df], ignore_index=True)

        return df
    
    except Exception as e:
        print(f"Error during augmentation: {e}")
        return df
    
def augment_word_bert(df, classes_to_augment, augmentation_percentage, text_column, model_path, random_state=42, weights=[0.7, 0.3]):
    try:
        eh.check_df_input(df, text_column)
        eh.check_classes_to_augment(classes_to_augment)
        eh.check_augmentation_percentage(augmentation_percentage)
        eh.check_random_state(random_state)
        eh.check_weights(weights, 2)
        
        aug_insert = naw.ContextualWordEmbsAug(model_path=model_path, action="insert")
        aug_substitute = naw.ContextualWordEmbsAug(model_path=model_path, action="substitute")

        augmented_rows = []

        for label in classes_to_augment:
            class_df = df[df['label'] == label]
            if class_df.empty:
                print(f"No data found for label '{label}'")
                continue
            num_samples_to_augment = int(len(class_df) * augmentation_percentage)
            sampled_df = class_df.sample(num_samples_to_augment, random_state=random_state)

            for _, row in sampled_df.iterrows():
                if not eh.handle_row_errors(row, text_column):
                    continue
                aug_choice = random.choices(
                    [aug_insert, aug_substitute],
                    weights=weights,
                    k=1
                )[0]
                translated_text = translate(row[text_column], aug=aug_choice)
                augmented_rows.append({text_column: translated_text, 'label': row['label']})

        augmented_df = pd.DataFrame(augmented_rows)
        df = pd.concat([df, augmented_df], ignore_index=True)

        return df
    except Exception as e:
        print(f"Error during augmentation: {e}")
        return df