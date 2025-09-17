import os
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
gpt_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
gpt_deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
gpt_model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
gpt_api_key = os.environ.get("AZURE_OPENAI_API_KEY")

azure_text_embedding_api_key = os.environ["AZURE_OPENAI_TEXT_EMBEDDING_API_KEY"]
azure_text_embedding_deployment_name = os.environ.get(
    "AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME"
)
azure_text_embedding_version = os.environ.get("AZURE_OPENAI_TEXT_EMBEDDING_API_VERSION")
azure_text_embedding_endpoint = os.environ.get("AZURE_SERVICE_ENDPOINT")

pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

llm = AzureChatOpenAI(
    api_version=gpt_api_version,
    azure_deployment=gpt_deployment_name,
    model_name=gpt_model_name,
    api_key=gpt_api_key,
)
embeddings = AzureOpenAIEmbeddings(
    api_key=azure_text_embedding_api_key,
    azure_deployment=azure_text_embedding_deployment_name,
    azure_endpoint=azure_text_embedding_endpoint,
    api_version=azure_text_embedding_version,
)

if __name__ == "__main__":

    # Get prompt from LangSmithHub
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # Connect Vector Store
    vectorstore = PineconeVectorStore(
        index_name=pinecone_index_name,
        embedding=embeddings,
    )

    retrival_chain = create_retrieval_chain(
        # Transform question 'what is Pinecone in machine learning?' into a vector and search for related documents in vector store.
        retriever=vectorstore.as_retriever(),
        # Take the documents found from the retriever step and combine them with the user's question, forming a single prompt to send to the LLM.
        combine_docs_chain=combine_docs_chain,
    )

    result = retrival_chain.invoke(
        input={"input": "what is Pinecone in machine learning?"}
    )

    print(result)

"""
{
  "input": "what is Pinecone in machine learning?",
  "answer": "Pinecone is a vector database solution that is used in machine learning to store and manage unstructured data, 
    such as text, images, or audio, in high-dimensional vectors. It allows for efficient indexing and querying of vector embeddings, 
    which is important for applications like similarity search, clustering, and classification within ML projects. 
    Pinecone provides features that enhance scalability, performance, and easy data retrieval for large-scale machine learning 
    applications, particularly in the context of projects involving large language models (LLMs).",
  "context": [
    {
      "id": "0ae24715-2707-4ea9-9340-1c4c591f2630",
      "metadata": {
        "source": "/Users/24729980/Documents/LangChain_examples_for_beginners/02_RAG/documents/vectordb_mediumblog.txt"
      },
      "page_content": "What are Embeddings?\nAn embedding is a technique for representing complex data, such as images, text, or audio, as numerical vectors.\nThese embeddings capture the essence of the data and show clearly the semantic similarity (or relationship) between different objects, with similar objects having vectors that are close to each other in the vector space. Thus, ML algorithms allow them to be efficiently processed and analyzed.\n\nML models often generate embeddings as part of their training process. For LLMs, an embedding model is put in place to create the embeddings\n\nEmbeddings are vectors that represent the essential features of a data point. For example, a natural language processing model might generate embeddings for words or sentences.\nEmbeddings can be used for a variety of tasks, such as clustering, classification, and anomaly detection. Vector databases can be used to store and query embeddings efficiently, which makes them ideal for ML applications."
    },
    {
      "id": "1010b7e1-680b-462b-aaed-083ccc57b8e8",
      "metadata": {
        "source": "/Users/24729980/Documents/LangChain_examples_for_beginners/02_RAG/documents/vectordb_mediumblog.txt"
      },
      "page_content": "Integration with ML Algorithms: Vector databases can be integrated with machine learning algorithms. This makes it easy to use vector databases to train and evaluate machine learning models. For example, you can use a vector database to store the data that is used to train a model, and then use the vector database to search for the data that is most relevant to the model.\nHandling Vector Embeddings: Vector databases provide a superior solution for handling vector embeddings by addressing the limitations of standalone vector indices, such as scalability challenges, cumbersome integration processes, and the absence of real-time updates and built-in security measures."
    },
    {
      "id": "1cec6fd9-5f9a-467d-8d10-f87df1f7f3c2",
      "metadata": {
        "source": "/Users/24729980/Documents/LangChain_examples_for_beginners/02_RAG/documents/vectordb_mediumblog.txt"
      },
      "page_content": "To see what embeddings look like, check out this Vectorizer created by Kenny, it converts texts to embeddings.\n\nNote: An embedding is a vector representation, but not all vectors are embeddings.\nPutting it all together, let us define a vector database.\n\nWhat is a Vector Database?\nA vector database is a type of database that stores and manages unstructured data, such as text, images, or audio, in high-dimensional vectors, to make it easy to find and retrieve similar objects quickly at scale in production.\nThey work by using algorithms like vector similarity search to index and query vector embeddings,\n\nThe importance of vector databases in LLM projects lies in their ability to provide easy search, high performance, scalability, and data retrieval by comparing values and finding similarities between them"
    },
    {
      "id": "80733a46-4e64-4db5-96ca-7e6981ea6521",
      "metadata": {
        "source": "/Users/24729980/Documents/LangChain_examples_for_beginners/02_RAG/documents/vectordb_mediumblog.txt"
      },
      "page_content": "List of Some Top Vector Databases\nThere are several vector database solutions available in the market, each with its own set of features and capabilities. Some of the top vector database solutions include: Weaviate, Pinecone, Chroma DB, Qdrant, Milvus\n\nHow To Choose The Right Vector Database For Your LLM Projects\nTo choose the right vector database for LLM projects, there are some factors you should consider. They include:\n\nScalability: Since LLMs generate and consume vast amounts of vector data, it is best to choose a database that can efficiently store and manage large-scale datasets without compromising performance. Also, the vector database must be able to seamlessly handle future data additions and expansion of your LLM project’s scope.\n\nPerformance: It should deliver fast query execution and swift retrieval of relevant vectors. It should also efficiently handle multi-dimensional queries and complex similarity searches."
    }
  ],
}
"""
