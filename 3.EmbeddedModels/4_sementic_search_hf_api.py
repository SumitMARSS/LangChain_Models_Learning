from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")

document = [
    "Virat Kohli is widely regarded as one of the best batsmen in modern cricket. Known for his aggressive batting style and remarkable consistency across formats, he has broken numerous records, including being one of the fastest players to score thousands of runs in ODIs. His leadership as captain helped India achieve several historic wins, both at home and abroad. Beyond numbers, Kohli’s passion and fitness have inspired a new generation of cricketers.",
    "Jasprit Bumrah has revolutionized fast bowling for India with his unique action and deadly yorkers. Making his mark first in limited-overs cricket, he soon became a reliable strike bowler in Test matches too. Bumrah’s ability to bowl accurate yorkers at the death makes him one of the most feared bowlers in T20 and ODI cricket. Despite injuries, he has made strong comebacks, proving his resilience and importance to the Indian team.",
    "Suresh Raina is often remembered as one of India’s finest middle-order batsmen in white-ball cricket. His ability to score quickly, especially in the last overs, made him a reliable finisher. Raina was also the first Indian to score a century in all three formats of the game. Apart from his batting, his electric fielding saved countless runs for India, and his contribution to Chennai Super Kings in the IPL made him a fan favorite.",
    "MS Dhoni, fondly called Captain Cool, is considered one of the greatest captains in cricket history. Under his leadership, India won all major ICC trophies, including the 2007 T20 World Cup, the 2011 ODI World Cup, and the 2013 Champions Trophy. Known for his calmness under pressure, Dhoni was a master finisher who often guided India to thrilling victories. His sharp wicketkeeping and leadership skills left a lasting impact on Indian cricket.",
    "Ravindra Jadeja is known for his all-round capabilities, contributing significantly with both bat and ball. As a left-arm spinner, he has been effective in containing runs and taking crucial wickets. Jadeja’s aggressive batting style allows him to change the course of the game quickly, especially in the lower order. His exceptional fielding skills make him one of the best fielders in the world, often saving vital runs and taking spectacular catches.",
    "Rohit Sharma, also known as the 'Hitman,' is renowned for his elegant batting style and ability to score big hundreds. He holds the record for the highest individual score in ODIs, with a monumental 264 runs. Rohit’s knack for converting fifties into hundreds has been a key asset for India in limited-overs cricket. As a captain, he has led Mumbai Indians to multiple IPL titles, showcasing his leadership qualities.",
]

query = "Tell me about Ravindra Jadeja ?"

vectorDocument = embeddings.embed_documents(document)
vectorQuery = embeddings.embed_query(query)

scores = cosine_similarity([vectorQuery], vectorDocument)[0]
index, value =  sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]

print(f"Best matched document (score: {value*100:.4f}):\n{document[index]}")
