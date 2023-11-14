import os
from typing import Any, Dict, List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

from langchain.vectorstores import Pinecone
import pinecone

PINE_CONE_API_KEY = os.environ.get("PINE_CONE_API_KEY", "YOUR_VECTOR_DB_INDEX")
PINE_ENVIRONMENT_REGION = os.environ.get("PINE_ENVIRONMENT_REGION", "YOUR_VECTOR_DB_REGION")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPEN_API_KEY")
INDEX_NAME = "medium-blogs-embeddings-index"

pinecone.init(api_key=PINE_CONE_API_KEY, environment=PINE_ENVIRONMENT_REGION)

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
  embeddings = OpenAIEmbeddings(open_api_key = OPENAI_API_KEY)
  docsearch = Pinecone.from_existing_index(
    embedding = embeddings,
    index_name = INDEX_NAME
  )

  chat_llm = ChatOpenAI(verbose=True, temperature=0)

  # qa = RetrievalQA.from_chain_type(
  #   llm=chat_llm,
  #   chain_type="stuff",
  #   retriever=docsearch.as_retriever(),
  #   return_source_documents=True
  # )

  # To maintain chat history or context or history chat, we should use ConversationalRetrievalChain
  qa = ConversationalRetrievalChain.from_llm(
    llm=chat_llm,
    retriever=docsearch.as_retriever(),
    return_source_documents=True
  )

  # return qa({"query": query})
  return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
  # print(run_llm(query="what is langchain?"))
  print(run_llm(query="who created langchain?"))


# """
# LangChain is a custom Large Language Model (LLM) that is tailored for organizations. 
# It is a technology that is used to automate customer service and enhance the 
# customer-company relationship. 
# LangChain is designed to provide efficient, personalized, and responsive interactions with 
# customers. It is capable of understanding and responding to customer queries and concerns, 
# and it can be integrated into various industries and organizations.
# """
