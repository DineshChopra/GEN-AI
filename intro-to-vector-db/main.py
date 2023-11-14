import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
from langchain.chains import RetrievalQA

import pinecone

PINE_CONE_API_KEY = os.environ.get("PINE_CONE_API_KEY", "YOUR_VECTOR_DB_INDEX")
PINE_ENVIRONMENT_REGION = os.environ.get("PINE_ENVIRONMENT_REGION", "VECTOR_DB_REGION")
pinecone.init(api_key=PINE_CONE_API_KEY, environment=PINE_ENVIRONMENT_REGION)

if __name__ == "__main__":
  print("Hello vector store")
  
  # Load File
  loader = TextLoader("intro-to-vector-db/mediumblogs/blog1.txt")
  documents = loader.load()
  
  # Split document into splits
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.split_documents(documents)
  print(len(texts))

  # Create embeddings
  embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"))
  docsearch = Pinecone.from_documents(texts, embeddings, index_name="medium-blogs-embeddings-index")

  # Create chain of query
  qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff",
    retriever=docsearch.as_retriever(), 
    return_source_documents=True
  )
  query = "what is vector database? give me 15 words answer for a bigenner"
  result=qa({"query": query})
  print(result)




