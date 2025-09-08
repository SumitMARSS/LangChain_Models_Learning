from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file


# tempreature is a parameter that controls the randomness of the model's output. A lower temperature (e.g., 0.2) makes the output more focused and deterministic, while a higher temperature (e.g., 0.8) makes it more random and creative.
######################  important -> each time i will get same output for same input because temperature is set to 0 .

# max_tokens is a parameter that sets the maximum number of tokens (words or word pieces) that the model can generate in its response. This helps to limit the length of the output.

# max_completion_tokens is a parameter that specifically limits the number of tokens in the completion part of the response, ensuring that the generated text does not exceed a certain length.
# Here, we are using the ChatOpenAI class to create a chat model instance with specific parameters.

# difference between max_tokens and max_completion_tokens is that max_tokens refers to the total token limit for the entire response, while max_completion_tokens specifically limits the tokens generated in the completion part of the response.


model = ChatOpenAI(model="gpt-4o", temperature=0.3, max_completion_tokens=256)
response = model.invoke(" Capital of india is ?")
print(response.content)