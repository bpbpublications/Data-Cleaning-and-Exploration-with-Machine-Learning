from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI

# Connect GPT-4 to external document retrieval
retriever = FAISS.load_local("my_vector_store")
qa_chain = RetrievalQA(llm=OpenAI(model="gpt-4"), retriever=retriever)

# Example query
response = qa_chain.run("Summarize the latest tax regulations.")
print(response)
