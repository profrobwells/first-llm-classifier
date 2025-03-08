# Evaluation

Before the advent of large-language models, machine-learning systems were trained using a technique called [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning). This approach required users to provide carefully prepared training data that showed the computer what was expected.

For instance, if you were developing a model to distinguish spam emails from legitimate ones, you would need to provide the model with a set of spam emails and another set of legitimate emails. The model would then use that data to learn the patterns and relationships between the inputs and outputs.

In addition to training the model, the curated input would be used to evaluate its performance. This process typically involved splitting the supervised data into two sets: one for training and one for testing. The model could then be evaluated using a separate set of supervised data to determine how well it performed by ensuring it could generalize to new data, not just memorize the examples it had been fed during training.

Large-language models operate differently. They are trained on vast amounts of text and can generate responses based on the patterns and relationships they learn through various approaches. The result is that they can be used to perform a wide range of tasks without requiring supervised data to be prepared beforehand.

This is a significant advantage. However, it also raises questions about evaluating an LLM prompt’s performance. If we don’t test its results, how do we know if it’s doing a good job? How can we improve if we can’t see where it gets things wrong?

In the final chapters, we will show how traditional supervision can still play a vital role in evaluating and improving an LLM prompt.

Start by outputting a random sample from the dataset to a file of comma-separated values. It will serve as our supervised sample. In general, the larger the sample, the better the evaluation, but at a certain point, the returns diminish. For this exercise, we will use a sample of 250 records.

```python
df.sample(250).to_csv("./sample.csv", index=False)
```

Now, you can download the file and inspect it in a spreadsheet program like Excel or Google Sheets. For each payee in the sample, you provide the correct category in a companion column. These decisions are then used to evalute the LLM's performance.

![Sample](_static/sample.png)

To speed the class along, we've already prepared a sample for you in [the class repository](https://github.com/palewire/first-llm-classifier). Our next step is to read it back into a DataFrame.

```python
sample_df = pd.read_csv("https://raw.githubusercontent.com/palewire/first-llm-classifier/refs/heads/main/_notebooks/sample.csv")
```

We need to install a bunch of packages.

```
!pip install groq rich ipywidgets retry pandas scikit-learn matplotlib seaborn
```

There's one very important tool ...


{emphasize-lines="6"}
```python
import json
from rich import print
from groq import Groq
from retry import retry
import pandas as pd
from sklearn.model_selection import train_test_split
```


```python
training_input, test_input, training_output, test_output = train_test_split(
    sample_df[['payee']],
    sample_df['category'],
    test_size=0.33,
    random_state=42,
)
```

Give the test set to the LLM.

```python
llm_df = classify_batches(list(test_input.payee))
```

```python
llm_df.category.value_counts()
```

{emphasize-lines="6-7,9"}
```python
import json
from rich import print
from groq import Groq
from retry import retry
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
```

```python
print(classification_report(
    test_output,
    llm_df.category,
    zero_division=False,
))
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">              precision    recall  f1-score   support

         Bar       <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.00</span>      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.00</span>      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.00</span>         <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>
       Hotel       <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.89</span>      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.80</span>      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.84</span>        <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>
       Other       <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.96</span>      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.96</span>      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.96</span>        <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">57</span>
  Restaurant       <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.87</span>      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.93</span>      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.90</span>        <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>

    accuracy                           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.94</span>        <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">83</span>
   macro avg       <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.93</span>      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.92</span>      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.93</span>        <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">83</span>
weighted avg       <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.94</span>      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.94</span>      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.94</span>        <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">83</span>

</pre>

```python
conf_mat = confusion_matrix(test_output, llm_df.category, labels=llm_df.category.unique())
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(
    conf_mat,
    annot=True,
    fmt='d',
    xticklabels=llm_df.category.unique(),
    yticklabels=llm_df.category.unique()
)
plt.ylabel('Actual')
plt.xlabel('Predicted')
```

![confusion matrix](_static/matrix-llm.png)

Now lets compare those results against the "old school" approach where the supervised sample is used to train machine-learning model that doesn't have access to the vast corpus of knowledge crammed into an LLM.

{emphasize-lines="10-13"}
```python
import json
from rich import print
from groq import Groq
from retry import retry
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
```

```python
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    min_df=5,
    norm='l2',
    encoding='latin-1',
    ngram_range=(1, 3),
)
preprocessor = ColumnTransformer(
    transformers=[
        ('payee', vectorizer, 'payee')
    ],
    sparse_threshold=0,
    remainder='drop'
)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LinearSVC(dual="auto"))
])
```

```python
model = pipeline.fit(training_input, training_output)
```

```python
predictions = model.predict(test_input)
```

Get a classification report in

Get a confusion report in
