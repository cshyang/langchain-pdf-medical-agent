from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.agents import (
    AgentExecutor,
    LLMSingleActionAgent,
)
from langchain.chat_models import ChatOpenAI
from tools import search_tool
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
import streamlit as st
from dotenv import load_dotenv
import prompt_template
import tools


def main():
    # create llm
    load_dotenv()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm_extract = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template.prompt)

    # Create tools
    tool_names = [tool.name for tool in tools.tools]

    # Create agent
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=prompt_template.output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools.tools, verbose=True
    )

    data_extaction_chain = LLMChain(
        llm=llm_extract, prompt=prompt_template.data_extaction_prompt, verbose=True
    )
    data_summmary_chain = LLMChain(
        llm=llm, prompt=prompt_template.medical_summary_prompt, verbose=True
    )

    # Streamtlit frontend
    st.set_page_config(page_title="Extract Biomarkers from PDF")
    st.header("Extract BioMarkers from PDF.")
    pdf = st.file_uploader("Upload your PDF.", type="pdf")

    if pdf:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        with get_openai_callback() as callback:
            data = data_extaction_chain.run(text)
            response = data_summmary_chain.run(data)
            print(callback)
        st.write(data)
        st.write(response)

    # with st.expander("Document Similarity Search"):


if __name__ == "__main__":
    main()
