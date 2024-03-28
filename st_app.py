import streamlit as st
from rag import *

st.set_page_config(page_title="LLM Search Titaninc", page_icon=':robot:')
# st.header("Query PDF")
st.title("Welcome")

# prompt = st.chat_input("Enter your message...")

if ('messages' not in st.session_state):
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Enter your message...")

if (prompt):
    st.session_state.messages.append({'role':'user', 'content': prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Processing"):
        # respond = inference(prompt)
        # with st.chat_message('assistant'):
        #     st.markdown(respond)

        with st.chat_message('assistant'):
            respond = st.write_stream(inference(prompt))
    
    st.session_state.messages.append({'role':'ai', 'content': respond})

    # print(st.session_state.messages)

# reset_button_key = "reset_button"
def reset_conversation():
  st.session_state.conversation = None
  st.session_state.messages = []
st.button('Reset Chat', on_click=reset_conversation)
