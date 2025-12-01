from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone

# Step 1: Connect to financial data
vector_store = Pinecone(index_name="finance-data", api_key="your-pinecone-api-key")

# Step 2: Build the RAG pipeline
retrieval_qa = RetrievalQA.from_chain_type(
    llm=OpenAI(model="gpt-4"),
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# Step 3: Query the pipeline
query = "Summarize the latest trends in the tech stock market."
response = retrieval_qa.run(query)

print("Market Summary:", response["answer"])
print("Sources:", response["source_documents"])
