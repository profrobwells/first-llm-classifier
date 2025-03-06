# Prompting with Python

Now that you've got your Python environment set up, it's time to start writing prompts and getting responses from an LLM. We'll be using the Groq API to interact with a model, but the concepts here will apply to other LLM providers like OpenAI or Anthropic.

First, we'll install the libraries we need: groq and rich.

```python
!pip install groq rich ipywidgets
```

Import

```python
from rich import print
from groq import Groq
```

Remember saving your Groq API key? Good, you'll need it now. Copy it from that text file and paste it inside the quotemarks.

```python
api_key = "Paste your key here"
```

Login to Groq and save the client for reuse

```python
client = Groq(api_key=api_key)
```

Let's make our first prompt. To do that, the `role` is "user" and the `content` is our prompt. We also need to pick a model from among the choices Groq gives us. We're picking Llama 3.3, the latest from Meta. It's pretty competitive with commercial models such as ChatGPT 4 and Claude 3.

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

Our client saves the response, and now we print that Python object to see what it contains.

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

There's a lot here, but the `message` has the actual response from the LLM. Let's just print the content from that message. Note that your response probably looks a little different from this guide. That's because LLMs mostly are probablistic prediction machines, not fact machines.

```python
print(response.choices[0].message.content)
```

```
Data journalism plays a crucial role in holding those in power accountable by providing fact-based insights and
analysis, enabling informed decision-making, and promoting transparency through the use of data-driven
storytelling.
```

Let's pick a different model from among [the choices that Groq offers](https://console.groq.com/docs/models). One we could try is Gemma2, an open model from Google.

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

Again, your response might vary from what's here. Let's find out.

```python
print(response.choices[0].message.content)
```

```
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


A well-structured prompt helps the LLM provide more accurate and useful responses. For instance, you can set a system prompt to establish the model's tone and role. Let's switch back to Llama 3.3 and provide a `system` message that provides a specific motivation for the LLM's responses.

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

Check out the results.

```python
print(response.choices[0].message.content)
```

```
Data journalism revolutionizes the way we consume news by using data analysis and visualization to uncover hidden
patterns, expose truth, and hold those in power accountable, making it an indispensable tool for a transparent and
informed society.
```

Want to see how tone affects the response? Change the system prompt to something old-school.

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

Then re-run the code and summon J. Jonah Jameson.

```python
print(response.choices[0].message.content)
```

```
If I must: data journalism is supposedly important because it allows reporters to use numbers and statistics to
uncover trends and patterns that might otherwise go unreported, but I still don't see the point of wasting good ink
on a bunch of soulless spreadsheets.
```
