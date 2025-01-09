import streamlit as st
import os
import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from graph_structure import SummarizerBot

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

###################################################### UI
st.set_page_config(page_title="SummarizerAI", page_icon='📝')
st.title("📝 Document Summarizer!")
st.subheader("", divider='rainbow')

###################################################### Chatbot usage
# Function to reset the session state
def reset_session():
    for key in st.session_state.keys():
        del st.session_state[key]

with st.sidebar:
    st.button("Start Over", on_click=reset_session, use_container_width=True)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
bot = SummarizerBot(model, embedder)

uploaded_file = st.file_uploader("Upload your document of any size here", type=["pdf"], accept_multiple_files=False)

if uploaded_file:
    with st.spinner("Summarizing ..."):
        # PyPDFLoader expects a file path, so save the uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            response = bot.builder.invoke({"file_path": uploaded_file.name})
            summary = response['summary']
            def stream_data():
                for word in summary.split(" "):
                    yield word + " "
                    time.sleep(0.03)
            st.write_stream(stream_data)
            
            # After being used, the uploaded file will be deleted.
            os.remove(uploaded_file.name)

        except Exception as e:
            st.markdown("Something went wrong. Please try again!")