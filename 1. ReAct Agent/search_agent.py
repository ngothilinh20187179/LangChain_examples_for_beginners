from dotenv import load_dotenv
import os
load_dotenv()

from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_tavily import TavilySearch

tools = [TavilySearch(tavily_api_key=os.environ["TAVILY_API_KEY"])]
llm = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    model=os.environ.get("AZURE_OPENAI_MODEL_NAME"),
)
prompt_template = hub.pull("hwchase17/react") # get prompt from langsmith hub or copy that content and paste into here
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools, 
    verbose=True
)

def main():
    res = agent_executor.invoke(
        input={
            "input": "search for a few articles on deep learning and cite the first 2"
        }
    )
    print(res)
    
if __name__ == "__main__":
    main()