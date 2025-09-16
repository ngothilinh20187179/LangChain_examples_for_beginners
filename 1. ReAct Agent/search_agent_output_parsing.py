from dotenv import load_dotenv
load_dotenv()
import os
from typing import List
from pydantic import Field, BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor

################
class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")
    
class AgentResponseWithSources(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's anwser to the query")
    source: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer"
    )

REACT_PROMPT_HWCHASE17 = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question formatted according to format_instructions: {format_instructions}

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


################
tools = [TavilySearch(tavily_api_key=os.environ["TAVILY_API_KEY"])]
llm = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    model=os.environ.get("AZURE_OPENAI_MODEL_NAME"),
)

# Khi dùng hub.pull("hwchase17/react"), agent tự động điền các input_variables
# Nhưng khi dùng REACT_PROMPT_HWCHASE17, sẽ lỗi khi cố điền format_instructions
# -> cần định nghĩa format_instructions
output_parser = PydanticOutputParser(pydantic_object=AgentResponseWithSources)
prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad", "format_instructions"],
    template=REACT_PROMPT_HWCHASE17,
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

def main():
    result = agent_executor.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
        }
    )
    print(result)

if __name__ == "__main__":
    main()