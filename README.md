# Summarize Your Long Document
This application returns a summary of your document of any size.

## Overview
Summerizer is a prototype AI application powered by `gpt-4o-mini` as its main model, and `text-embedding-3-large` as its embedding model, both provided by **OpenAI**. The framework that has been used for building this application is [LangGraph](https://www.langchain.com/langgraph), a stateful, orchestration framework that brings added control to the workflows. The scaffolding and all other abstractions are provided by [LangChain](https://www.langchain.com/), a powerful open-source framework that facilitates the integration of LLMs into applications. For demonstration purposes, the app utilizes Streamlit's open-source app framework.

## Installation
The programming language used for building the app is Python.
```python
pip install -r requirements.txt
```

## Usage
After installing the dependencies, an OpenAI API key is required. Users must assign their own OpenAI API key to `OPENAI_API_KEY` variable in `.env` file on the project's root directory. Then running the application is straightforward:
```python
streamlit run app.py
```
This will turn the code into a web app, executing on user's default browser.