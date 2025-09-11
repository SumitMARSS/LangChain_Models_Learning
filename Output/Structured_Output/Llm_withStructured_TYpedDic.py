from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm)


# annotated TypedDict for mapping the output to a structured format as per requirements
# optional fields can be marked using Optional from typing module
# Literal can be used to restrict the values of a field to a specific set of strings/values
# schema definition

class Review(TypedDict): 
    summary: Annotated[str, "A concise summary of the review in 1-2 sentences."] 
    sentiment: Annotated[Literal["pos", "neg"], "The overall sentiment of the review, either 'positive', 'negative', or 'neutral'."]
    # as ollama does not support structured output

    # key_themes: Annotated[list[str], "A list of key themes discussed in the review."]
    # pros : Annotated[Optional[list[str]], "A list of positive aspects mentioned in the review."]
    # cons : Annotated[Optional[list[str]], "A list of negative aspects mentioned in the review."]
 

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """Christopher Nolan’s Inception is a cinematic masterpiece that blends science fiction, psychological thriller, and heist genres into one of the most intellectually stimulating films of the 21st century. From the very first scene, the movie immerses you in a world where dreams and reality seamlessly intertwine, leaving viewers questioning what is real and what isn’t until the final frame.
        The plot follows Dom Cobb, played brilliantly by Leonardo DiCaprio, a professional thief with the rare ability to infiltrate people’s subconscious and steal valuable secrets. What makes the story unique is the central concept of “inception” — planting an idea in someone’s mind without them realizing it. Cobb’s journey is not only about completing the seemingly impossible mission but also about confronting his own guilt, grief, and emotional turmoil surrounding his late wife, Mal, portrayed by Marion Cotillard.
        What truly stands out is Nolan’s meticulous attention to detail. The layered dream sequences, each governed by its own set of rules, are crafted with stunning precision. The famous “rotating hallway” fight scene and the gravity-defying visuals are not just spectacles; they serve the story’s logic and heighten the sense of disorientation the characters face.
        The ensemble cast — including Joseph Gordon-Levitt, Ellen Page, Tom Hardy, and Cillian Murphy — delivers performances that balance intelligence, wit, and emotional depth. Hans Zimmer’s thunderous yet haunting score adds a sense of urgency and wonder, particularly the iconic “Time” theme that lingers long after the credits roll.
        Ultimately, Inception is more than just a heist movie set inside dreams. It’s a story about love, loss, and the power of ideas — how they can shape reality and consume our consciousness. The ambiguous ending, with the spinning top, ensures the film stays alive in our minds, provoking discussions and theories for years to come.
    """
)
print(result)
print(type(result))
print()
print("Summary ->",result['summary'])
print("Sentiment ->", result['sentiment'])
