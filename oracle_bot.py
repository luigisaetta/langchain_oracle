#
# Streamlit App to demo OCI AI GenAI
# this is the main code, with the UI
#
import streamlit as st

# this function initialise the rag chain, creating retriever, llm and chain
from init_rag_streamlit import initialize_rag_chain, get_answer

#
# Configs
#


#
# Main
#
st.title("OCI Generative AI Bot powered by RAG")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# init RAG
rag_chain = initialize_rag_chain()
print("RAG chain Ready!!!")
print()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if question := st.chat_input("Hello, how can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    # here we call OCI genai...
    print("...")
    response = get_answer(rag_chain, question)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
