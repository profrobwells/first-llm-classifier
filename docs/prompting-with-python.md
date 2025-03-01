# Prompting with Python

Install the libraries we need.

```python
!pip install groq rich ipywidgets
```

Import

```python
from rich import print
from groq import Groq
```

Set the API key

```python
api_key = "Paste your key here"
```

Login to Groq and save the client for reuse

```python
client = Groq(api_key=api_key)
```

Make our first prompt

```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence",
        }
    ],
    model="llama-3.3-70b-versatile",
)
```

Print the response

```python
print(response)
```

```python
ChatCompletion(
    id='chatcmpl-e219e15c-471f-468c-a0f7-69ba31c83da6',
    choices=[
        Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content='Data journalism plays a crucial role in holding those in power accountable by providing
fact-based insights and analysis, enabling informed decision-making, and promoting transparency through the use of
data-driven storytelling.',
                role='assistant',
                function_call=None,
                reasoning=None,
                tool_calls=None
            )
        )
    ],
    created=1740671812,
    model='llama-3.3-70b-versatile',
    object='chat.completion',
    system_fingerprint='fp_76dc6cf67d',
    usage=CompletionUsage(
        completion_tokens=37,
        prompt_tokens=46,
        total_tokens=83,
        completion_time=0.134545455,
        prompt_time=0.00492856,
        queue_time=0.231341476,
        total_time=0.139474015
    ),
    x_groq={'id': 'req_01jn4200h0e4s8e12pj5d2e3ye'}
)
```

Pull out the text we actually want

```python
print(response.choices[0].message.content)
```

```plaintext
Data journalism plays a crucial role in holding those in power accountable by providing fact-based insights and
analysis, enabling informed decision-making, and promoting transparency through the use of data-driven
storytelling.
```

Substitute in a different model for comparison. Link to Groq https://console.groq.com/docs/models

{emphasize-lines="8"}
```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence",
        }
    ],
    model="gemma2-9b-it",
)
```

```python
print(response.choices[0].message.content)
```

```plaintext
Data journalism illuminates complex issues, empowers informed decision-making, and drives accountability through
the rigorous analysis and visualization of data.
```

:::{admonition} Sidenote
Groq's Python library is very similar to the ones offered by OpenAI, Anthropic and other LLM providers. If you prefer to use those tools, the techniques you learn here should be easily transferable.

For instance, here's how you'd make this same call with Anthropic's Python library:

```python
from anthropic import Anthropic

client = Anthropic(api_key=api_key)

response = client.messages.create(
    messages=[
        {"role": "user", "content": "Explain the importance of data journalism in a concise sentence"},
    ],
    model="claude-3-5-sonnet-20240620",
)

print(response.content[0].text)
```
:::

Show how you can make a system prompt to prime the LLM. Point out we switched back to Llama.

{emphasize-lines="3-7,13"}
```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "you are an enthusiastic nerd who believes data journalism is the future."

        },
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence",
        }
    ],
    model="llama-3.3-70b-versatile",
)
```

```python
print(response.choices[0].message.content)
```

```plaintext
Data journalism revolutionizes the way we consume news by using data analysis and visualization to uncover hidden
patterns, expose truth, and hold those in power accountable, making it an indispensable tool for a transparent and
informed society.
```

Change the system prompt.

{emphasize-lines="3-7"}
```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "you are a crusty, ill-tempered editor who hates math and thinks data journalism is a waste of time and resources."

        },
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence",
        }
    ],
    model="llama-3.3-70b-versatile",
)
```

```python
print(response.choices[0].message.content)
```

```plaintext
If I must: data journalism is supposedly important because it allows reporters to use numbers and statistics to
uncover trends and patterns that might otherwise go unreported, but I still don't see the point of wasting good ink
on a bunch of soulless spreadsheets.
```
