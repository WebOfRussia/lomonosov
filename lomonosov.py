import pandas as pd
import streamlit as st
from app.common import system_prompt
from app.common.tools import SearchPaperTool, PDFReaderTool, BibtexGeneratorTool
from langchain.agents import (
    AgentExecutor,
    create_gigachat_functions_agent,
)
from langchain.agents.gigachat_functions_agent.base import (
    format_to_gigachat_function_messages,
)
from langchain_community.chat_models import GigaChat
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

if os.getenv("environment") != "production":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(BASE_DIR, '.env')
    load_dotenv(dotenv_path)

# Авторизация в сервисе GigaChat
AUTH_DATA = os.getenv("AUTH_DATA")

llm = GigaChat(
    credentials=AUTH_DATA,
    verify_ssl_certs=False,
    timeout=600,
    model='GigaChat-Pro-preview',
    scope='GIGACHAT_API_CORP',
    temperature=0.1
    # messages=[SystemMessage(content=system_prompt)]

)

tools = [SearchPaperTool(), PDFReaderTool()]
agent = create_gigachat_functions_agent(llm, tools)

# AgentExecutor создает среду, в которой будет работать агент
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=False, return_intermediate_steps=True
)

st.set_page_config(page_title="ИИ Ассистент \"Ломоносов\"")
st.title("ИИ Ассистент \"Ломоносов\"")

st.markdown(
    """
    ℹ️ Я ищу ответы на ваши вопросы в [КиберЛенинке](https://cyberleninka.ru/) (но скоро дойду и до других...)
    """
)


"""
-----
"""

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 400px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.image(os.path.join("resources", "img", "logo.jpeg"))
    st.markdown("""Примеры команд:""")
    st.code("Найди мне статьи по теме:")
    st.code("Прочитай статью №")
    # вопрос по статье
    st.code("Какие основные выводы?")
    st.code("Сгенерируй Bibtex и верни его в markdown")
    st.code("Выведи e-mail авторов")
    st.markdown("""[Пример статьи](https://cyberleninka.ru/article/n/uvelichenie-tochnosti-bolshih-yazykovyh-modeley-s-pomoschyu-rasshirennoy-poiskovoy-generatsii/pdf)""")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Привет, я Ломоносов!"}]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content=system_prompt)]

if "metadata" not in st.session_state:
    st.session_state.metadata = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Обратитесь ко мне..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    with st.spinner('Пошёл в Москву узнавать...'):
        with st.chat_message("assistant"):
            # запуск агента
            print(str(st.session_state.chat_history))
            result = agent_executor.invoke(
                {
                    "chat_history": st.session_state.chat_history,
                    "input": prompt + ". Ищи данные в диалоге в первую очередь.",
                }
            )

            if type(result["output"]) == dict:
                response = result["output"]["markdown"]
                st.session_state.metadata = result["output"]["metadata"]
                st.session_state.chat_history.append(HumanMessage(content=str(st.session_state.metadata)))
            else:
                response = result["output"]

            # обновление истории диалога
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history += format_to_gigachat_function_messages(result["intermediate_steps"])
            st.session_state.chat_history.append(AIMessage(content=response))
            
            st.markdown(response, unsafe_allow_html=True)        

    st.session_state.messages.append({"role": "assistant", "content": response})
