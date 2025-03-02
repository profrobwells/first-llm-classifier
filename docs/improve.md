# Improving prompts

The other good thing about coding the supervised sample is that you can use the training slice of the set as a few-shot example for the LLM. This is a great way to improve the performance of the model. You can also use the test set to evaluate the performance of the model.

```python
def get_fewshots(traning_input, training_ouput, batch_size=10):
    input_batches = get_batch_list(list(training_input.payee), n=batch_size)
    output_batches = get_batch_list(list(training_output), n=batch_size)
    fewshot_list = []
    for i, input_list in enumerate(input_batches):
        fewshot_list.extend([
            {
                "role": "user",
                "content": "\n".join(input_list),
            },
            {
                "role": "assistant",
                "content": json.dumps(output_batches[i])
            }
        ])
    return fewshot_list
```

```python
fewshot_list = get_fewshots(input_batches, output_batches)
```

```python
fewshot_list[:2]
```

{emphasize-lines="27-39,42"}
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
    messages = [
        {
            "role": "system",
            "content": prompt,
        }
    ]
    messages += fewshot_list
    messages += [
        {
            "role": "user",
            "content": "\n".join(name_list),
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
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

Now run it again

```python
llm_df = classify_batches(list(test_input.payee))
```

And see if your results are any better

```python
print(classification_report(
    test_output,
    llm_df.category,
    zero_division=False,
))
```

Another good idea is to find the most common misclassifications and tweak your prompt to address any patterns they reveal.

```python
proof_df = llm_df.merge(
    sample_df,
    on="payee",
    how="inner",
    suffixes=["_llm", "_human"]
)
```

```python
proof_df[proof_df.category_llm != proof_df.category_human]
```

```python
proof_df.head()
```

{emphasize-lines="23"}
```python
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

If a business name contains both the word "Restaurant" and the word "Bar", you should classify it as a Restaurant.

Ensure that the number of classifications in your output matches the number of business names in the input. It is very important that the length of JSON list you return is exactly the same as the number of business names your receive.
"""
```
