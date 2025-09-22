# do some imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

#make a few tools
#First one generates a question using an llm
@tool
def generate_question():
    """Generates a question"""
    response =llm.invoke("Generate a question")
    return response.content


#Tool 2: Answers the question
@tool
def answer_question(question: str):
    """Answers the question"""
    response = llm.invoke(question)
    return response.content


#hook the tools
agent_question_asker = create_react_agent(llm, [generate_question]);
agent_question_answerer = create_react_agent(llm, [answer_question]);

def extract_last_ai_content(state):
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return message.content
    return ""

#invoke the agents with a clean messages structure
res_1 = agent_question_asker.invoke({
    "messages": [
        ("user", "Generate a single interesting question. Use the generate_question tool."),
    ]
})

question = extract_last_ai_content(res_1) or "Generate a single interesting question."

res_2 = agent_question_answerer.invoke({
    "messages": [
        ("user", question),
    ]
})

print("Generated question:", question)
print("Answer:", extract_last_ai_content(res_2))