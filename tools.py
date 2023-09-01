from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun


def search_tool(input_text):
    search = DuckDuckGoSearchRun()
    search_results = search.run(input_text)
    return search_results


tools = [
    Tool(
        name="search medlineplus",
        func=search_tool,
        description="useful for when you need to answer questions about current events",
    )
]
