import pandas as pd
from easy_text_augmenter import augment_random_word

df = pd.DataFrame({
    'processed_text': ['This is a test', 'Another test', 'More data', 'Sample text', 'Hello world I am human','Goodbye world I am robot'],
    'label': ['A', 'A', 'B', 'B', 'A', 'B']
})
classes_to_augment = ['A', 'B']
augmented_df = augment_random_word(df, classes_to_augment, augmentation_percentage=0.5, text_column='processed_text')
print(augmented_df)
