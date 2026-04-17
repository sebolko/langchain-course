from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

taviily = TavilyClient()

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
tools = [search]
agent = create_agent(model=llm, tools=tools)

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