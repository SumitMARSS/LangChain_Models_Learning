from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os, json


load_dotenv()
# Initialize client with a free model
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=api_key)
# Define a simple JSON schema manually
schema = {
    "name": "Review",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Short summary of the review"},
            "sentiment": {"type": "string", "description": "Sentiment: positive, negative, or neutral"},
            "reviewer": {"type": "string", "description": "Name of the reviewer"}
        },
        "required": ["summary", "sentiment", "reviewer"]
    }
}

# Ask the model with JSON schema output
response = client.chat_completion(
    model="HuggingFaceH4/zephyr-7b-beta",
    messages=[
        {"role": "user", "content": "Write a JSON review of the movie Inception."}
    ],
    response_format={"type": "json_schema", "json_schema": schema},
    max_tokens=300,
)

# Extract JSON string
raw_output = response.choices[0].message["content"]

print("Raw output (string):", raw_output)

# Convert string to Python dict
try:
    review_data = json.loads(raw_output)
    print("\n✅ JSON parsed successfully:")
    print(review_data)
    print("Summary ->", review_data["summary"])
    print("Sentiment ->", review_data["sentiment"])
    print("Reviewer ->", review_data["reviewer"])
except Exception as e:
    print("\n⚠️ Failed to parse JSON:", e)


