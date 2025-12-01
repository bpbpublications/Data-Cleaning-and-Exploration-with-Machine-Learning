from cachetools import TTLCache
import asyncio
import openai
import os

# Initialize a TTL cache for frequently used queries
cache = TTLCache(maxsize=100, ttl=600)

os.environ["OPENAI_API_KEY"] = "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"

async def generate_response(query):
    # Check cache
    if query in cache:
        return cache[query]
    
    # Retrieve relevant documents (simulated retrieval)
    retrieved_docs = ["FAQ content 1", "FAQ content 2"]

    # Combine documents and query
    prompt = f"Answer the question: '{query}' using the documents: {retrieved_docs}"

    # Call OpenAI's API asynchronously
    response = await openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=200
    )

    # Cache and return the response
    cache[query] = response.choices[0].message.content
    return response.choices[0].message.content

# Example query
query = "What is the return policy?"
response = asyncio.run(generate_response(query))
print(response)
