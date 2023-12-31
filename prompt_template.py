from langchain.agents import (
    Tool,
    AgentOutputParser,
)
from langchain.prompts import StringPromptTemplate, PromptTemplate
from langchain.schema import AgentAction, AgentFinish
import re
from typing import List, Union
import tools

template = """ Give a short summary of the input biomarkers and give recommendations, but speaking as compasionate medical professional.

Use the following format:

Question: the input questions you must answer
Thought: you should always think about what to do
Action: the action to take.
Action Input: the input to the action you wish
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer the final answer to the original input question

Begin! Remember to ask if want to schedule a tele health consultation with a medical professional.

Biomarkers: {input}
{agent_scratchpad}

"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools=tools.tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"],
)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


output_parser = CustomOutputParser()

data_extaction_prompt = PromptTemplate(
    input_variables=["medical-report"],
    template="Extract only the biomarkers and the assosiated value from {medical-report}. Put the result into table relational format with 4 columns: biomarker, value, unit, and abnormal status. ",
)

medical_summary_prompt = PromptTemplate(
    input_variables=["biomarker-results"],
    template="""
    Analyze and give a short summary of the abnormal {biomarker-results} and give recommendations, 
    but speaking as compasionate medical professional. 
    Remember to ask if want to schedule a tele health consultation with a medical professional if needed.
    
    """,
)
