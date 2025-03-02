# Structured responses

You don't have to ask for essays, poems or chitchat. You can ask an LLM to make very simple decisions and code data.

```python
prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.
"""
```

{emphasize-lines="5"}
```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": prompt
        },
    ],
    model="llama-3.3-70b-versatile",
)
```

{emphasize-lines="7-10"}
```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": "Minnesota Twins",
        }
    ],
    model="llama-3.3-70b-versatile",
)
```

```python
print(response.choices[0].message.content)
```

```plaintext
Major League Baseball (MLB)
```

{emphasize-lines="9"}
```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": "Minnesota Vikings",
        }
    ],
    model="llama-3.3-70b-versatile",
)
```

```python
print(response.choices[0].message.content)
```

```plaintext
National Football League (NFL)
```

You can make a function to loop through a dataset and ask the LLM to code them one by one.

```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": name,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return response.choices[0].message.content
```

```python
team_list = ["Minnesota Twins", "Minnesota Vikings", "Minnesota Timberwolves"]
```

```python
for team in team_list:
    league = classify_team(team)
    print(team, league)
```

```python
['Minnesota Twins', 'Major League Baseball (MLB)']
['Minnesota Vikings', 'National Football League (NFL)']
['Minnesota Timberwolves', 'National Basketball Association (NBA)']
```

Sometimes the LLM will get weird and return something you don't want. You can improve this be adding validation.

{emphasize-lines="9-12,31-37"}
```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.

Your responses must come from the following list:
- Major League Baseball (MLB)
- National Football League (NFL)
- National Basketball Association (NBA)
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": name,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    answer = response.choices[0].message.content

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
    ]
    if answer not in acceptable_answers:
        raise ValueError(f"{answer} not in list of acceptable answers")

    return answer
```

```python
classify_team("Indiana Fever")
```

```python
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[47], line 1
----> 1 classify_team("Indiana Fever")

Cell In[45], line 36, in classify_team(name)
     30 acceptable_answers = [
     31     "Major League Baseball (MLB)",
     32     "National Football League (NFL)",
     33     "National Basketball Association (NBA)",
     34 ]
     35 if answer not in acceptable_answers:
---> 36     raise ValueError(f"{answer} not in list of acceptable answers")
     38 return answer

ValueError: Women's National Basketball Association (WNBA)

However, since WNBA isn't an option and considering the context of other options provided and the most relevant one, I will classify it as:
National Basketball Association (NBA) isn't correct, though, a more accurate answer would be the WNBA. not in list of acceptable answers
```

Sometimes there just isn't answer in your validation list. One way to manage that is to allow an "other" category.

{emphasize-lines="14,37"}
```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.

Your responses must come from the following list:
- Major League Baseball (MLB)
- National Football League (NFL)
- National Basketball Association (NBA)

If the team's league is not on the list, you should label them as "Other".
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": name,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    answer = response.choices[0].message.content

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
        "Other",
    ]
    if answer not in acceptable_answers:
        raise ValueError(f"{answer} not in list of acceptable answers")

    return answer
```

```python
classify_team("Indiana Fever")
```

```python
'Other'
```

Sometimes the LLM just gets too creative. For poems and essays, this is great. For structured responses, not so much. You can dial it down and ask the LLM to be straightforward and consistent by reducing its temperature setting to zero.


{emphasize-lines="29"}
```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.

Your responses must come from the following list:
- Major League Baseball (MLB)
- National Football League (NFL)
- National Basketball Association (NBA)

If the team's league is not on the list, you should label them as "Other".
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": name,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer = response.choices[0].message.content

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
        "Other",
    ]
    if answer not in acceptable_answers:
        raise ValueError(f"{answer} not in list of acceptable answers")

    return answer
```

You can also increase reliability by priming the LLM with examples of the type of response you want. This technique is called "few-shot prompting".

{emphasize-lines="23-54"}
```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.

Your responses must come from the following list:
- Major League Baseball (MLB)
- National Football League (NFL)
- National Basketball Association (NBA)

If the team's league is not on the list, you should label them as "Other".
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "Los Angeles Rams",
            },
            {
                "role": "assistant",
                "content": "National Football League (NFL)",
            },
            {
                "role": "user",
                "content": "Los Angeles Dodgers",
            },
            {
                "role": "assistant",
                "content": " Major League Baseball (MLB)",
            },
            {
                "role": "user",
                "content": "Los Angeles Lakers",
            },
            {
                "role": "assistant",
                "content": "National Basketball Association (NBA)",
            },
            {
                "role": "user",
                "content": "Los Angeles Kings",
            },
            {
                "role": "assistant",
                "content": "Other",
            },
            {
                "role": "user",
                "content": name,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer = response.choices[0].message.content

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
        "Other",
    ]
    if answer not in acceptable_answers:
        raise ValueError(f"{answer} not in list of acceptable answers")

    return answer
```

You can also ask the function to automatically retry if it doesn't get a valid response.

```
!pip install groq rich ipywidgets retry
```

{emphasize-lines="3"}
```python
from rich import print
from groq import Groq
from retry import retry
```

{emphasize-lines="1"}
```python
@retry(ValueError, tries=2, delay=2)
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.
...
```
