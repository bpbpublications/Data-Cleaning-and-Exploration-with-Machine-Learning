from openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Step 1: Build a vector store for educational content
course_materials = [
    {"title": "Introduction to Python", "content": "Python is a versatile programming language."},
    {"title": "Data Structures", "content": "Understanding arrays, lists, and dictionaries."},
]


# Convert dicts to Document objects for FAISS
course_documents = [
    Document(page_content=course["content"], metadata=course)
    for course in course_materials
]

embeddings = OpenAIEmbeddings(api_key="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")
vector_store = FAISS.from_documents(course_documents, embeddings)

# Step 2: Initialize the RAG pipeline
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model="gpt-4"),
    retriever=vector_store.as_retriever()
)

# Step 3: Query the pipeline
query = "Can you recommend resources to learn Python basics?"
response = retrieval_chain.run(query)

print("Recommendation:", response)
