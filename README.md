# Easy Text Augmenter

Easy Text Augmenter is a Python package for augmenting text data directly on your pandas dataframe using various NLP techniques. 
There are only 3 techniques for now : 
- augment_random_word
- augment_random_character
- augment_word_bert

## Installation

```bash
!pip install easy-nlp-augmentation
```


## How to use
### augment_random_word
```python
import pandas as pd
from easy_text_augmenter import augment_random_word

df = pd.DataFrame({
    'processed_text': ['This is a test', 'Another test data ', 'Of course we need more data', 'Newton does not like apple', 'Hello world I am a human'],
    'label': ['A', 'A', 'B', 'B', 'A']
})
classes_to_augment = ['A', 'B']
augmented_df = augment_random_word(df, classes_to_augment, augmentation_percentage=0.8, text_column='processed_text')
print(augmented_df)
```
**Result :**

```
                          text label
0               This is a test     A
1           Another test data      A
2  Of course we need more data     B
3   Newton does not like apple     B
4     Hello world I am a human     A
5             Th is is a te st     A
6                 Another data     A
7   Does not newton like apple     B
```

### augment_random_character
```python
from easy_text_augmenter import augment_random_word

augmented_df = augment_random_character(df, classes_to_augment, augmentation_percentage=0.8, text_column='text')
print(augmented_df)
```
**Result :**

```
                          text label
0               This is a test     A
1           Another test data      A
2  Of course we need more data     B
3   Newton does not like apple     B
4     Hello world I am a human     A
5               This is a estt     A
6            Another te8t data     A
7   Newtun d0e8 not like apple     B
```


### augment_word_bert
```python
from easy_text_augmenter import augment_word_bert

augmented_df = augment_word_bert(df, classes_to_augment, augmentation_percentage=0.5, text_column='processed_text', model_path='bert-base-uncased', random_state=70)
print(augmented_df)
```
**Result :**

```
                                processed_text label
0                               This is a test     A
1                           Another test data      A
2                  Of course we need more data     B
3                   Newton does not like apple     B
4                     Hello world I am a human     A
5                         another test of data     A
6                      this term is not a test     A
7  newton does absolutely not like every apple     B
```

## Authors

Contact me at :
 
- [shizuka0@proton.me](mailto:shizuka0@proton.me)
- [shizuka.my.id](https://shizuka.my.id/)

## Documentation

<details>
<summary>augment_random_word</summary>

### augment_random_word
**Description:**

The `augment_random_word` function augments a specified percentage of samples in given classes of a DataFrame by randomly applying one of three augmentation techniques (swap, delete, split) to the text column.

`
augment_random_word(df, classes_to_augment, augmentation_percentage, text_column, random_state=42, weights=[0.5, 0.3, 0.2])
`

**Parameters:**
- `df` (pandas.DataFrame): The input DataFrame containing the text data and labels.
- `classes_to_augment` (list): A list of class labels that need to be augmented.
- `augmentation_percentage` (float): The percentage of samples to augment from each specified class.
- `text_column` (str): The name of the column in the DataFrame that contains the text data.
- `random_state` (int, optional): A random seed for reproducibility. Default is 42.
- `weights` (list, optional): A list of weights to determine the probability of selecting each augmentation type. Default is [0.5, 0.3, 0.2] for swap, delete, and split, respectively.

`weights` techniques :
- swap: randomly swap word in text.
- delete: randomly delete word in text.
- split: randomly split word in text.

**Returns:**
- pandas.DataFrame: A new DataFrame with the augmented data appended to the original data.

</details>


<details>
<summary>augment_random_character</summary>

### augment_random_character
**Description:**

The `augment_random_character` function performs random character-based augmentations on specific classes of text data within a DataFrame. It uses several augmentation techniques to randomly alter characters in the text, increasing the diversity of the dataset.

`
augment_random_character(df, classes_to_augment, augmentation_percentage, text_column, random_state=42, weights=[0.2, 0.2, 0.2, 0.2, 0.2])
`

**Parameters:**
- `df` (pd.DataFrame): The input DataFrame containing text data and their corresponding labels.
- `classes_to_augment` (list): A list of class labels indicating which classes should be augmented.
- `augmentation_percentage` (float): The percentage of samples in each class that should be augmented.
- `text_column` (str): The column name in the DataFrame that contains the text data to be augmented.
- `random_state` (int, optional): The seed used by the random number generator for reproducibility. Default is 42.
- `weights` (list, optional): A list of weights for each augmentation technique, used to determine the probability of choosing each technique. Default is [0.2, 0.2, 0.2, 0.2, 0.2].

`weights` techniques :
- aug_ocr: OCR-based augmentation.
- aug_keyboard: Keyboard error simulation.
- aug_insert: Random character insertion.
- aug_swap: Random character swapping.
- aug_delete: Random character deletion.

**Returns:**
- pandas.DataFrame: A new DataFrame with the augmented data appended to the original data.

</details>

<details>
<summary>augment_word_bert</summary>

### augment_word_bert
**Description:**

The `augment_word_bert` function augments text data in a DataFrame using a BERT-based word augmentation technique. It inserts or substitutes words in the specified text column for a given percentage of samples in the specified classes.

`
def augment_word_bert(df, classes_to_augment, augmentation_percentage, text_column, model_path, random_state=42, weights=[0.7, 0.3])
`

**Parameters:**
- `df` (pandas.DataFrame): The DataFrame containing the data to be augmented.
- `classes_to_augment` (list): A list of class labels indicating which classes should be augmented.
- `augmentation_percentage` (float): The percentage of samples within each class to augment (e.g., 0.2 for 20%).
- `text_column` (str): The name of the column in the DataFrame that contains the text to be augmented.
- `model_path` (str): The path to the pre-trained BERT model used for augmentation.
- `random_state` (int, optional): The random seed used for sampling (default is 42).
- `weights` (list, optional): The weights for choosing between the insertion and substitution augmentation techniques (default is [0.7, 0.3]).

**Returns:**
- pandas.DataFrame: The original DataFrame with additional augmented samples.

</details>
