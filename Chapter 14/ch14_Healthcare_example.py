from vertexai.preview.language_models import Gemini
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone

# Step 1: Connect to a medical database
pinecone_api_key = "your-pinecone-api-key"
pinecone_env = "us-west1-gcp"
vector_store = Pinecone(index_name="medical-database", api_key=pinecone_api_key, environment=pinecone_env)

# Step 2: Initialize the RAG pipeline
llm = Gemini(model="gemini-health-v1")
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# Step 3: Query the pipeline
query = "What are the latest treatments for type 2 diabetes?"
response = retrieval_qa.run(query)

print("Treatment Recommendations:", response["answer"])
print("Sources:", response["source_documents"])
