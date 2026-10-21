import uuid

from dotenv import load_dotenv
import os
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import gradio as gr

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

def get_date():
    """Get the current date"""
    return datetime.now().strftime("%d/%m/%Y")

search_tool = TavilySearchResults()

conn = sqlite3.connect("chatbot_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

llm = ChatOllama(model="qwen2.5:3b")

system_prompt = """
You are a helpful assistant.
Answer all user's queries.
Use the get_date tool ONLY when the user is explicitly asking about the today's date.
Use the search_tool ONLY when
 - The user explicitly asks you to search the web
 - The question is about breaking news
 - The answer is likely to have changed very recently
Do not use search_tool for general programming, mathematics, science, or educational questions.
"""

agent = create_agent(
    model=llm,
    tools=[get_date,search_tool],
    system_prompt=system_prompt,
    checkpointer=checkpointer)

#print(response['messages'][-1].content)

def chat(message, history,thread_id):
    config = {"configurable":{"thread_id": thread_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config
    )
    last_response = response['messages'][-1].content
    return last_response

with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot")
    thread_id = gr.State(value=lambda: str(uuid.uuid4()))
    gr.ChatInterface(fn=chat, additional_inputs=[thread_id])

demo.launch(server_name="127.0.0.1", server_port=7861)