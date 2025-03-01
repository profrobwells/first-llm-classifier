# Bulk prompts

Requesting answers one by one can take a long time. And it can end up costing you money.

The solution is to submit your requests in batches and then get the answers back from the LLM as JSON you can parse.

{emphasize-lines="1"}
```python
import json
from rich import print
from groq import Groq
from retry import retry
```

{emphasize-lines="2,17-27,36-43,46,53-54,62-66"}
```python
@retry(ValueError, tries=2, delay=2)
def classify_teams(name_list):
    prompt = """
You are an AI model trained to classify text.

I will provide list of professional sports team names separated by new lines

You will reply with the sports league in which they compete.

Your responses must come from the following list:
- Major League Baseball (MLB)
- National Football League (NFL)
- National Basketball Association (NBA)

If the team's league is not on the list, you should label them as "Other".

Your answers should be returned as a flat JSON list.

It is very important that the length of JSON list you return is exactly the same as the number of names your receive.

If I were to submit:

"Los Angeles Rams\nLos Angeles Dodgers\nLos Angeles Lakers\nLos Angeles Kings"

You should return the following:

["National Football League (NFL)", "Major League Baseball (MLB)", "National Basketball Association (NBA)", "Other"]
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "Chicago Bears,Chicago Cubs,Chicago Bulls,Chicago Blackhawks",
            },
            {
                "role": "assistant",
                "content": '["National Football League (NFL)", "Major League Baseball (MLB)", "National Basketball Association (NBA)", "Other"]',
            },
            {
                "role": "user",
                "content": "\n".join(name_list),
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer_str = response.choices[0].message.content
    answer_list = json.loads(answer_str)

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
        "Other",
    ]
    for answer in answer_list:
        if answer not in acceptable_answers:
            raise ValueError(f"{answer} not in list of acceptable answers")

    return dict(zip(name_list, answer_list))
```

Try that with our team list

```python
classify_teams(team_list)
```

```python
{'Minnesota Twins': 'Major League Baseball (MLB)',
 'Minnesota Vikings': 'National Football League (NFL)',
 'Minnesota Timberwolves': 'National Basketball Association (NBA)'}
```

One common problem is that your number of inputs can fail to match your number of inputs when you have a large batch.

{emphasize-lines="66-69"}
```python
@retry(ValueError, tries=2, delay=2)
def classify_teams(name_list):
    prompt = """
You are an AI model trained to classify text.

I will provide list of professional sports team names separated by new lines

You will reply with the sports league in which they compete.

Your responses must come from the following list:
- Major League Baseball (MLB)
- National Football League (NFL)
- National Basketball Association (NBA)

If the team's league is not on the list, you should label them as "Other".

Your answers should be returned as a flat JSON list.

It is very important that the length of JSON list you return is exactly the same as the number of names your receive.

If I were to submit:

"Los Angeles Rams\nLos Angeles Dodgers\nLos Angeles Lakers\nLos Angeles Kings"

You should return the following:

["National Football League (NFL)", "Major League Baseball (MLB)", "National Basketball Association (NBA)", "Other"]
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "Chicago Bears,Chicago Cubs,Chicago Bulls,Chicago Blackhawks",
            },
            {
                "role": "assistant",
                "content": '["National Football League (NFL)", "Major League Baseball (MLB)", "National Basketball Association (NBA)", "Other"]',
            },
            {
                "role": "user",
                "content": "\n".join(name_list),
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer_str = response.choices[0].message.content
    answer_list = json.loads(answer_str)

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
        "Other",
    ]
    for answer in answer_list:
        if answer not in acceptable_answers:
            raise ValueError(f"{answer} not in list of acceptable answers")

    try:
        assert len(name_list) == len(answer_list)
    except:
        raise ValueError(f"Number of outputs ({len(name_list)}) does not equal the number of inputs ({len(answer_list)})")

    return dict(zip(name_list, answer_list))
```


Now let's get some real data out.

```
!pip install groq rich retry pandas
```

{emphasize-lines="5"}
```python
import json
from rich import print
from groq import Groq
from retry import retry
import pandas as pd
```

```python
df = pd.read_csv("https://raw.githubusercontent.com/palewire/first-llm-classifier/refs/heads/main/_notebooks/Form460ScheduleESubItem.csv")
```

```python
df.sample(10)
```

Let's adapt what we have to match...

{emphasize-lines="2-26,33-48,61-66"}
```python
@retry(ValueError, tries=2, delay=2)
def classify_payees(name_list):
    prompt = """You are an AI model trained to categorize businesses based on their names.

You will be given a list of business names, each separated by a new line.

Your task is to analyze each name and classify it into one of the following categories: Restaurant, Bar, Hotel, or Other.

It is extremely critical that there is a corresponding category output for each business name provided as an input.

If a business does not clearly fall into Restaurant, Bar, or Hotel categories, you should classify it as "Other".

Even if the type of business is not immediately clear from the name, it is essential that you provide your best guess based on the information available to you. If you can't make a good guess, classify it as Other.

For example, if given the following input:

"Intercontinental Hotel\nPizza Hut\nCheers\nWelsh's Family Restaurant\nKTLA\nDirect Mailing"

Your output should be a JSON list in the following format:

["Hotel", "Restaurant", "Bar", "Restaurant", "Other", "Other"]

This means that you have classified "Intercontinental Hotel" as a Hotel, "Pizza Hut" as a Restaurant, "Cheers" as a Bar, "Welsh's Family Restaurant" as a Restaurant, and both "KTLA" and "Direct Mailing" as Other.

Ensure that the number of classifications in your output matches the number of business names in the input. It is very important that the length of JSON list you return is exactly the same as the number of business names your receive.
"""
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "Intercontinental Hotel\nPizza Hut\nCheers\nWelsh's Family Restaurant\nKTLA\nDirect Mailing",
            },
            {
                "role": "assistant",
                "content": '["Hotel", "Restaurant", "Bar", "Restaurant", "Other", "Other"]',
            },
            {
                "role": "user",
                "content": "Subway Sandwiches\nRuth Chris Steakhouse\nPolitical Consulting Co\nThe Lamb's Club",
            },
            {
                "role": "assistant",
                "content": '["Restaurant", "Restaurant", "Other", "Bar"]',
            },
            {
                "role": "user",
                "content": "\n".join(name_list),
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer_str = response.choices[0].message.content
    answer_list = json.loads(answer_str)

    acceptable_answers = [
        "Restaurant",
        "Bar",
        "Hotel",
        "Other",
    ]
    for answer in answer_list:
        if answer not in acceptable_answers:
            raise ValueError(f"{answer} not in list of acceptable answers")

    try:
        assert len(name_list) == len(answer_list)
    except:
        raise ValueError(f"Number of outputs ({len(name_list)}) does not equal the number of inputs ({len(answer_list)})")

    return dict(zip(name_list, answer_list))
```

```python
sample_list = list(df.sample(10).payee)
```

```python
classify_payees(sample_list)
```

```python
... example output here ...
```

Break out bigger dataset into batches

```python
def get_batch_list(li, n=10):
    batch_list = []
    for i in range(0, len(li), n):
        batch_list.append(li[i : i + n])
    return batch_list
```

Write a function that will split up a big list and classify it batch by batch.

{emphasize-lines="1,4"}
```python
import time
import json
from rich import print
from rich.progress import track
from groq import Groq
from retry import retry
import pandas as pd
```


```python
def classify_batches(name_list, batch_size=10, wait=2):
    # Store the results
    all_results = {}

    # Batch up the list
    batch_list = get_batch_list(name_list, n=batch_size)

    # Loop through the list in batches
    for batch in track(batch_list):
        # Classify it
        batch_results = classify_payees(batch)
        # Add it to the results
        all_results.update(batch_results)
        # Tap the brakes
        time.sleep(wait)

    # Return the results
    return all_results
```

```python
bigger_sample = list(df.sample(100).payee)
```

```python
classify_batches(bigger_sample)
```

We can make the result a dataframe

{emphasize-lines="18-21"}
```python
def classify_batches(name_list, batch_size=10, wait=2):
    # Store the results
    all_results = {}

    # Batch up the list
    batch_list = get_batch_list(name_list, n=batch_size)

    # Loop through the list in batches
    for batch in track(batch_list):
        # Classify it
        batch_results = classify_payees(batch)
        # Add it to the results
        all_results.update(batch_results)
        # Tap the brakes
        time.sleep(wait)

    # Return the results
    return pd.DataFrame(
        all_results.items(),
        columns=["payee", "category"]
    )
```

```python
results_df = classify_batches(bigger_sample)
```

```python
results_df.head()
```

```python
results_df.value_counts()
```
