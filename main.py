from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient

from typing import List
from pydantic import BaseModel, Field

load_dotenv()

taviily = TavilyClient()

class Source(BaseModel):
  """Schema for a source used by the agent"""

  url:str =Field(description="The URL of the source")

class AgentResponse(BaseModel):
  """Schema for the agent's response"""

  answer: str = Field(description="The answer to the user's query")
  sources: List[Source] = Field(default_factory=list, description="The sources used to answer the query")

@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet.
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for: {query}")
    return taviily.search(query=query)

llm = ChatOpenAI(model="gpt-5", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain course!")
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo?"}
        ]
    })
    print(result)

if __name__ == "__main__":
    main()