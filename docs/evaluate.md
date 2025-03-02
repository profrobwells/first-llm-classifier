# Evaluation

Explain how traditional ML and all this stuff has historically been trained and evaluated with "supervised learning" and how this is not the case with LLMs. Point out that we can still evaluate LLMs using the same tools. And then compare how this approach compares to the "old school" approach.

Output a sample.

```python
df.sample(250).to_csv("./sample.csv", index=False)
```

Show a screenshot of how I manually coded them.

Now we read it back in.

```python
sample_df = pd.read_csv("./sample.csv")
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
