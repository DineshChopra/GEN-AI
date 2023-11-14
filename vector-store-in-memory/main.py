import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LOCAL_VECTOR_DB_NAME = "faiss_index_react"

if __name__ == "__main__":
  # print("OPENAI_API_KEY --- ", OPENAI_API_KEY)
  pdf_path = "vector-store-in-memory/2210.03629.pdf"
  loader = PyPDFLoader(file_path=pdf_path)
  documents = loader.load()

  # control the chunk size
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
  docs = text_splitter.split_documents(documents=documents)

  # convert chunk into embeddings and store into local vector store 
  embeddings = OpenAIEmbeddings()
  vector_store = FAISS.from_documents(docs, embeddings)
  vector_store.save_local(LOCAL_VECTOR_DB_NAME)

  new_vectorstore = FAISS.load_local(LOCAL_VECTOR_DB_NAME, embeddings)
  qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())
  query = "Give me the gist of React in 3 sentences"
  result = qa.run(query)
  print(result)



