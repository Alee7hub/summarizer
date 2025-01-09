# Warning control
import warnings
warnings.filterwarnings('ignore')

import ast
import numpy as np
from sklearn.cluster import KMeans
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class OverallState(TypedDict):
    file_path: str
    pages: list
    metadata: str
    summary: str

class SummarizerBot:
    def __init__(self, model, embedder):
        builder = StateGraph(OverallState)
        builder.add_node("loader", self.loader)
        builder.add_node("doc_metadata", self.metadata_extractor)
        builder.add_node("summarizer", self.summarizer)
        builder.add_edge("loader", "doc_metadata")
        builder.add_edge("doc_metadata", "summarizer")
        builder.add_edge("summarizer", END)
        builder.set_entry_point("loader")
        self.model = model
        self.embedder = embedder
        self.builder = builder.compile()
    
    def loader(self, state: OverallState):
        FILE = state["file_path"]
        loader = PyPDFLoader(FILE)
        pages = loader.load()
        state["pages"] = pages
        return state
    
    def metadata_extractor(self, state: OverallState):
        first_page = state["pages"][0]
        first_page_content = first_page.page_content
        first_page_metadata = first_page.metadata
        template = """
        You will be provided with the content and metadata associated with the first page of a document. Your task is to identify and extract the three to five most important pieces of metadata for the entire document. When performing this task, follow these guidelines:

        1. Exclude Page-Specific Information: Do not include metadata related to page numbers.
        2. Enrich Metadata: If the provided metadata is overly simple or lacks key details, enrich it by inferring additional relevant information based on the content of the first page.
        3. Fallback Option: If the first page's content does not provide any useful insights, use the provided first page metadata as is.
        4. Formatting: Ensure the response strictly follows the format of the provided metadata.

        Example Response Format:
        {{'title': 'War_and_Peace.pdf', 'author': 'John Doe', 'year': 1850, 'publication': 'Muster', ...}}

        Input Details:
        First page's content: {content}
        First page's metadata: {metadata}
        """
        prompt = ChatPromptTemplate.from_template(template=template)
        chain = prompt | self.model
        result = chain.invoke({"content": first_page_content, "metadata": first_page_metadata}).content
        state["metadata"] = result
        return state
    
    def summarizer(self, state: OverallState):
        metadata = ast.literal_eval(state["metadata"])
        pages = state["pages"]
        vectors = self.embedder.embed_documents([page.page_content for page in pages])
        vectors_np = np.array(vectors)

        # Combine the pages, and replace the tabs with spaces
        text = ""
        for page in pages:
            text += page.page_content
        text = text.replace('\t', ' ')
        num_tokens = self.model.get_num_tokens(text)

        if num_tokens < 100000:
            num_clusters = 10
        else:
            num_clusters = int(10 + (num_tokens / 100000))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors_np)

        # Find the closest embeddings to the centroids

        # Create an empty list that will hold your closest points
        closest_indices = []

        # Loop through the number of clusters you have
        for i in range(num_clusters):
            
            # Get the list of distances from that particular cluster center
            distances = np.linalg.norm(vectors_np - kmeans.cluster_centers_[i], axis=1)
            
            # Find the list position of the closest one (using argmin to find the smallest distance)
            closest_index = np.argmin(distances)
            
            # Append that position to your closest indices list
            closest_indices.append(closest_index)
        
        selected_indices = [numpy_arr.tolist() for numpy_arr in sorted(closest_indices)]
        selected_docs = [pages[doc] for doc in selected_indices]

        map_template = """
        You will be given a section of a book. This section will be enclosed in triple backticks (```)
        Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
        Your response should be at least one paragraph and fully encompass what was said in the passage.

        ```{text}```
        FULL SUMMARY:
        """
        map_prompt = ChatPromptTemplate.from_messages(
            [("human", map_template)]
        )

        reduce_template = """
        The following is a set of summaries:
        {docs}
        Take these and distill it into a final, consolidated summary
        of the main themes. Highlight the important notes as bulletpoints.
        Also use the following information to give the reader a background 
        about the document in the beginning of your final summary:
        {metadata}
        """
        reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

        parser = StrOutputParser()
        map_chain = map_prompt | self.model | parser
        reduce_chain = reduce_prompt | self.model | parser

        # Make an empty list to hold your summaries
        summary_list = []
        # Loop through a range of the lenght of your selected docs
        for i, doc in enumerate(selected_docs):
            # Go get a summary of the chunk
            chunk_summary = map_chain.invoke({"text": doc})
            # Append that summary to your list
            summary_list.append(chunk_summary)

        summaries = "\n".join(summary_list)

        # Convert it back to a document
        summaries = Document(page_content=summaries)

        output = reduce_chain.invoke({"docs": summaries, "metadata": metadata})
        state['summary'] = output
        return state