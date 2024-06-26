import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac 
import random

def translate(text, aug):
    translated = aug.augment(text)
    return translated[0]

def augment_random_word(df, classes_to_augment, augmentation_percentage, text_column, random_state=42, weights=[0.5, 0.3, 0.2]):
    aug_swap = naw.RandomWordAug(action="swap")
    aug_del = naw.RandomWordAug(action="delete")
    aug_split = naw.SplitAug()

    augmented_rows = []

    for label in classes_to_augment:
        class_df = df[df['label'] == label]
        num_samples_to_augment = int(len(class_df) * augmentation_percentage)
        sampled_df = class_df.sample(num_samples_to_augment, random_state=random_state)

        for _, row in sampled_df.iterrows():
            aug_choice = random.choices(
                [aug_swap, aug_del, aug_split],
                weights=weights,
                k=1
            )[0]
            translated_text = translate(row[text_column], aug=aug_choice)
            augmented_rows.append({text_column: translated_text, 'label': row['label']})

    augmented_df = pd.DataFrame(augmented_rows)
    df = pd.concat([df, augmented_df], ignore_index=True)

    return df

def augment_random_character(df, classes_to_augment, augmentation_percentage, text_column, random_state=42, weights=[0.2, 0.2, 0.2, 0.2, 0.2]):
    aug_ocr = nac.OcrAug()
    aug_keyboard = nac.KeyboardAug()
    aug_insert = nac.RandomCharAug(action="insert")
    aug_swap = nac.RandomCharAug(action="swap")
    aug_delete = nac.RandomCharAug(action="delete")

    augmented_rows = []

    for label in classes_to_augment:
        class_df = df[df['label'] == label]
        num_samples_to_augment = int(len(class_df) * augmentation_percentage)
        sampled_df = class_df.sample(num_samples_to_augment, random_state=random_state)

        for _, row in sampled_df.iterrows():
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

def augment_word_bert(df, classes_to_augment, augmentation_percentage, text_column, model_path, random_state=42, weights=[0.7, 0.3]):
    aug_insert = naw.ContextualWordEmbsAug(model_path=model_path, action="insert")
    aug_substitute = naw.ContextualWordEmbsAug(model_path=model_path, action="substitute")

    augmented_rows = []

    for label in classes_to_augment:
        class_df = df[df['label'] == label]
        num_samples_to_augment = int(len(class_df) * augmentation_percentage)
        sampled_df = class_df.sample(num_samples_to_augment, random_state=random_state)

        for _, row in sampled_df.iterrows():
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