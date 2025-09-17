# LangChain_examples_for_beginners

## Set up
- Dependency setup
    + pip install uv
    + uv init
    + uv add python-dotenv
    + uv add langchain

- Use LangSmith to trace
    + Sign up https://www.langchain.com/langsmith
    + Generate API key (in Set up tracing)
    + Paste info into .env 
        + LANGSMITH_TRACING=true
        + LANGSMITH_ENDPOINT=https://api.smith.langchain.com
        + LANGSMITH_API_KEY=your-api-key
        + LANGSMITH_PROJECT=your-pj-name

0. Summary text: use model from openai or ollama, use LangSmith to trace
- Install:
    + uv add langchain-openai
    + uv add langchain-ollama
- Download ollama and pull model: https://ollama.com/
- Paste api's informations into .env (endpoint, key, name, deploy-name, version,...)

1. ReAct Agent: 
- search_agent: use tavily search tool and use prompt template of LangSmith's hub
    - Install tavily or other web search engines (DuckDuckGo,...)
        + uv add langchain-tavily
        + uv add langchain_community
    - Create API Key (https://app.tavily.com/home) and paste it into .env (TAVILY_API_KEY)

2. RAG: ingest text document and retrieve
- Install pinecone, langchainhub
    + uv add langchain-pinecone 
    + uv add langchainhub
- Set up vectorstore pinecone
    + Sign up pinecone : https://app.pinecone.io/
    + Create new pinecone index (use model text-embedding-3-small)
    + Create new pinecone api key
    + Copy & Paste into env: PINECONE_INDEX_NAME, PINECONE_API_KEY
- Create new api for embedding use azureopenai(optional)
    + https://ai.azure.com/
    + Use model: text-embedding-3-small
    + Copy keys and paste into .env (endpoint, key, name, deploy-name, version,...)