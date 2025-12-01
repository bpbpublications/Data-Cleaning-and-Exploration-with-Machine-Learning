from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Step 1: Build a vector store for FAQs
faq_data = [
    {"question": "What is the return policy?", "answer": "You can return items within 30 days."},
    {"question": "How do I track my order?", "answer": "Use the tracking link sent via email."},
]
# Convert dicts to Document objects for FAISS
faq_documents = [
    Document(page_content=faq["question"] + "\n" + faq["answer"], metadata=faq)
    for faq in faq_data
]
embeddings = OpenAIEmbeddings(api_key="")
vector_store = FAISS.from_documents(faq_documents, embeddings)

# Step 2: Initialize the RAG pipeline
retrieval_qa = RetrievalQA.from_chain_type(
    llm=OpenAI(model="gpt-4"),
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# Step 3: Query the RAG pipeline
query = "Can I return a product after 20 days?"
response = retrieval_qa.run(query)

print("Response:", response["answer"])
print("Source Document:", response["source_documents"])
