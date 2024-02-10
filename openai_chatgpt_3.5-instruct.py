# Chat with an intelligent assistant in your terminal
from openai import OpenAI
from prompts.contexts import context_document, biography
from prompts.instructs import instructions
from libs.helper import num_tokens_from_string
from settings.open_ai_api import OPEN_AI_SECRET

welcome_message = "Hello, introduce yourself as Artificial Milad using 2-3 sentences."
client = OpenAI(api_key=OPEN_AI_SECRET,)#Read at https://platform.openai.com/docs/api-reference/chat/create

history = "role: system content: "+ f""" Answer technical work questions based on the following:\n {context_document} \n
    answer personal questions based on the following : \n {biography} \n 
    ### Instruction: {instructions} """ +"role: user, content: say " +  welcome_message + "Answer: ",


response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt = history,
)
print(response)

