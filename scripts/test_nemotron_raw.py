import os

from openai import OpenAI


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

response = client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    messages=[
        {
            "role": "user",
            "content": "Explain what an MLP is in the context of Transformers.",
        }
    ],
    temperature=0,
    max_tokens=4096,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True,
        },
        "reasoning_budget": 2048,
    },
)

message = response.choices[0].message

print("=" * 80)
print("REASONING")
print("=" * 80)
print(getattr(message, "reasoning_content", None))

print()
print("=" * 80)
print("FINAL ANSWER")
print("=" * 80)
print(message.content)

print()
print("=" * 80)
print("USAGE")
print("=" * 80)
print(response.usage)