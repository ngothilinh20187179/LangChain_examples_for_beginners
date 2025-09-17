# Load data from a file, split it, create vector embedding and store it in a Vector Store.
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

document_path = "/Users/24729980/Documents/LangChain_examples_for_beginners/02_RAG/documents/vectordb_mediumblog.txt"
azure_text_embedding_api_key = os.environ["AZURE_OPENAI_TEXT_EMBEDDING_API_KEY"]
azure_text_embedding_deployment_name = os.environ.get(
    "AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME"
)
azure_text_embedding_version = os.environ.get("AZURE_OPENAI_TEXT_EMBEDDING_API_VERSION")
azure_text_embedding_endpoint = os.environ.get("AZURE_SERVICE_ENDPOINT")

pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

if __name__ == "__main__":
    # ingesting
    # loader = PyPDFLoader(file_path=pdf_path)
    loader = TextLoader(document_path)
    document = loader.load()

    # splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # embedding
    embeddings = AzureOpenAIEmbeddings(
        api_key=azure_text_embedding_api_key,
        azure_deployment=azure_text_embedding_deployment_name,
        azure_endpoint=azure_text_embedding_endpoint,
        api_version=azure_text_embedding_version,
    )

    # store
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=pinecone_index_name
    )
    print("finish")
