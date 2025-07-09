import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Load .env variables
load_dotenv()

def explain_preprocessing(summary_steps: list, dataset_name: str = "the uploaded dataset"):
    summary_text = "\n".join(f"- {step}" for step in summary_steps)

    # Define prompt template
    prompt_template = PromptTemplate(
        input_variables=["dataset_name", "summary"],
        template="""
You are a helpful data science assistant.

The user uploaded a dataset called "{dataset_name}". The system applied the following preprocessing steps:

{summary}

Write a clear, friendly explanation of what was done and why each step was necessary. Use simple bullet points and keep it concise.
"""
    )

    try:
        prompt = prompt_template.format(dataset_name=dataset_name, summary=summary_text)

        # Initialize LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

        messages = [
            SystemMessage(content="You are a data science assistant who explains preprocessing steps clearly."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    except Exception as e:
        return f"Error generating explanation: {str(e)}"
