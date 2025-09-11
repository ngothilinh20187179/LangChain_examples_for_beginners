# LangChain_examples_for_beginners

- Dependency setup
    + pip install uv
    + uv init
    + uv add python-dotenv
    + uv add langchain

- Use LangSmith to tracing
    + Sign up https://www.langchain.com/langsmith
    + Generate API key (in Set up tracing)
    + Paste info to .env 
        LANGSMITH_TRACING=true
        LANGSMITH_ENDPOINT=https://api.smith.langchain.com
        LANGSMITH_API_KEY=<your-api-key>
        LANGSMITH_PROJECT=<your-pj-name>

0. Summary text
- Install:
    + uv add langchain-openai
    + uv add langchain-ollama
- Download ollama and pull model: https://ollama.com/
- Past api's informations to .env (endpoint, key, name, deploy-name, version,...)
