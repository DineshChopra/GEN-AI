import os

from langchain.document_loaders import ReadTheDocsLoader, PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

PINE_CONE_API_KEY = os.environ.get("PINE_CONE_API_KEY", "YOUR_VECTOR_DB_INDEX")
PINE_ENVIRONMENT_REGION = os.environ.get("PINE_ENVIRONMENT_REGION", "VECTOR_DB_REGION")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPEN_API_KEY")

pinecone.init(api_key=PINE_CONE_API_KEY, environment=PINE_ENVIRONMENT_REGION)

def ingest_docs() -> None:
  loader = ReadTheDocsLoader(path="langchain-doc-helper/langchain-docs")
  raw_documents = loader.load()
  print(len(raw_documents))

def ingest_pdf_docs() -> None:
  path = "langchain-doc-helper/docs/"
  loader = PyPDFDirectoryLoader(path)
  raw_documents = loader.load()
  doc_sources = set()
  for doc in raw_documents:
    doc_sources.add(doc.metadata["source"])
  print(doc_sources)
  print(len(raw_documents))

  # Create chunks
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
  documents = text_splitter.split_documents(documents=raw_documents)
  print(f"Splitted into {len(documents)} chunks")

  # Create embeddings and upload to vector db
  embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
  # print(f"Going to add {len(documents)} to Pinecone, with {len(embeddings)} embeddings")
  Pinecone.from_documents(documents, embeddings, index_name="medium-blogs-embeddings-index")



if __name__ == "__main__":
  # ingest_docs()
  ingest_pdf_docs()
