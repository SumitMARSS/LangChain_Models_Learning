from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()  

model = ChatAnthropic(model="claude-2", temperature=0.3, max_completion_tokens=256)
response = model.invoke(" Capital of india is ?")
print(response.content)