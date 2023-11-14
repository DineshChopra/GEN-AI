Q: What is RAG (retrieval-augmented generation)
Ans: RAG is an AI framework for retrieving facts from an external knowledge base to ground large language models (LLMs) on the most accurate, up-to-date information and to give users insight into LLMs' generative process. Langchain is an example of RAG framework

Q:
Sparse Search (Exact match search), Dense Search(Semantic Search), Hybrid Search (Mix of sparse and dense search)

Q: Vector embedding => capture meaning => think => machine understandable format of the data

Q: How can we measure the distance between two vectors:
Ans: 
1. Euclidean Distance (L2)
2. Manhattan Distance (L1)
3. Dot Product
4. Cosine distance

1. Load all documents by using PyPDFDirectoryLoader
2. Create chunks by using `RecursiveCharacterTextSplitter`
```
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
```
3. Create embeddings
```
embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
```
4. Load into vector db `Pinecone`, `cromedb`, `weaviate`



Brute force search:



Model parallelisation
Ensembeling
Bagging and boosting

Embedding limitations:
How many type of embedding:

Scalable:

For semantic search: K nearest neighbours algorithm is used
For performance we used clustering, divide whole vector store into multiple clusters and then find results

### Prompt:
* Zero shot prompting: Model is guessing at its best effort without having seen any example of the result you want. Prompt does not contain any explicit instructions or examples for the model to follow. Instead it relies on the model’s ability to understand and interpret natural language.
* One shot prompting: Model is given just one example of the result you want.
* Few shot prompting: Model is given a few examples of the result you want. In this prompt we are providing task description plus few examples.



Reference:
Vectore database: https://www.deeplearning.ai/short-courses/
Langchain: https://www.udemy.com/course/langchain/


