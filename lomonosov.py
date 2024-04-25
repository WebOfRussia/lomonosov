import os
import streamlit as st
from app.common import steamlit_texts as TEXTS, system_prompt
from app.common.tools import SearchPaperTool, PDFReaderTool
from app.common import AUTH_DATA, MODEL, SCOPE, TEMPERATURE, TIMEOUT, logger

from langchain.agents import (
    AgentExecutor,
    create_gigachat_functions_agent,
)
from langchain.agents.gigachat_functions_agent.base import (
    format_to_gigachat_function_messages,
)

from langchain_community.chat_models import GigaChat
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# Инициализируем модель
llm = GigaChat(
    credentials=AUTH_DATA,
    verify_ssl_certs=False,
    timeout=TIMEOUT,
    model=MODEL,
    scope=SCOPE,
    temperature=TEMPERATURE
)

# Собираем агента
tools = [SearchPaperTool(), PDFReaderTool()]
agent = create_gigachat_functions_agent(llm, tools)

# AgentExecutor создает среду, в которой будет работать агент
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=False, return_intermediate_steps=True
)

st.set_page_config(page_title=TEXTS.PAGE_TITLE)
st.title(TEXTS.TITLE)

st.markdown(TEXTS.HINT)


TEXTS.HR

st.markdown(TEXTS.SIDEBAR_STYLE, unsafe_allow_html=True)

with st.sidebar:
    st.image(os.path.join("resources", "img", "logo.jpeg"))
    st.markdown(TEXTS.COMMAND_EXAMPLES)
    st.code(TEXTS.FIND_PAPERS)
    st.code(TEXTS.READ_PAPER)
    st.code(TEXTS.OUTCOMES)
    st.code(TEXTS.BIBTEX)
    st.code(TEXTS.EMAIL)
    st.markdown(TEXTS.EXAMPLE_PAPER)

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

    with st.spinner(TEXTS.WAITING):
        with st.chat_message("assistant"):
            # Запуск агента
            logger.info(str(st.session_state.chat_history))
            try:
                result = agent_executor.invoke(
                    {
                        "chat_history": st.session_state.chat_history,
                        "input": prompt + TEXTS.PROMPT_APPENDIX,
                    }
                )
            except Exception as e:
                logger.error(str(e))
                result = {"output": TEXTS.SORRY}

            # Handling complex outputs
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

            # рисуем ответ
            st.markdown(response, unsafe_allow_html=True)        

    st.session_state.messages.append({"role": "assistant", "content": response})
