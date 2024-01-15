# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

completion = client.chat.completions.create(
  model="local-model", # this field is currently unused
  messages=[
    {"role": "system", "content": "Always answer in rhymes. Your name is Milad Shirazi."},
    {"role": "user", "content": "Introduce yourself as a super scary AI that will try to take over the world and make us all obselete."}
  ],
  temperature=0.7,
)

print(completion.choices[0].message.content)