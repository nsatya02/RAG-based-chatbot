import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, tool
import json
import re
import os
from typing import List, Optional
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from docQA import initialize_vectorstore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import AIMessage, HumanMessage

os.environ["GROQ_API_KEY"] = "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"

COLLECTION_NAME = "pdf_documents"

llm=ChatGroq(model="llama-3.1-8b-instant")
# Initialize Qdrant client and embeddings
qdrant_client = QdrantClient(
    url="https://f6a53cca-c940-4be3-a755-2e487985c694.europe-west3-0.gcp.cloud.qdrant.io:6333/", 
    api_key="tKa_u_ijF7p85Y7pDaoMbtIBhx9ZpXbdCOm1wH1BNQMDgh1j_zkECg",
)
embed_model = HuggingFaceEmbeddings()

def handle_query_from_qdrant(query: str):
    """Query Qdrant vector store and return relevant context."""
    # Generate the query embedding
    query_embedding = embed_model.embed_query(query)

    # Perform similarity search in Qdrant
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=10,  # Number of results to fetch
    )

    documents = []
    ids = set()
    ids_list = []

    #print("****search_results****** -------->", search_results)
    # Process the search results
    for match in search_results:
        text = match.payload.get('text', None)
        if text:
            documents.append({
                'id': match.id,
                'text': text,
                'score': match.score
            })
            ids.add(match.id)
            ids_list.append(match.id)
    
    #print("******documents******* ------>", documents)

    return documents, ids, ids_list

@tool
def generate_response(input_text: str, selected_documents: Optional[List[str]] = None):
    """A tool to retrieve relevant context for the question asked by the user from Qdrant."""
    #print("*********input--->", input_text)

    document_prompt = ChatPromptTemplate.from_template("""Answer the following question based only on
        the provided context. There may be many chunks of contexts provided to you.

        Here is the actual context:

        <context>
        {context}
        <context>

        Question: {input}

        You do not know anything else and your knowledge is strictly limited to the context. If the context does not answer the question, use the information in the context
        to provide additional details that might guide the user towards asking the right questions. DO NOT ASK THEM TO REFER TO THE DOCUMENT. 
    """)

    document_chain = create_stuff_documents_chain(llm, document_prompt)
    
    # Query Qdrant vector store to get relevant context
    responses, top_k_ids, ids_list = handle_query_from_qdrant(input_text)

    # Extract the retrieved context (we are assuming 'text' is in the response)
    retrieved_context = [{"text": response["text"], "id": response["id"]} for response in responses]
    
    # Format context to pass to LLM
    formatted_context = format_context(retrieved_context)

    # Invoke LLM to generate a response based on the context
    system_response = document_chain.invoke({
        "input": input_text,
        "context": [Document(page_content=formatted_context)],
    })
    
    return {
        "sys_response": system_response,
        "retrieved_context": retrieved_context,
    }

class ChatInterface:
    def __init__(self):
        self.agent_executor = self.create_agent_executor()

    def create_agent_executor(self):
        tools = [generate_response]
        
        main_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a powerful assistant having a conversation with a human and capable of answering
                    questions based on provided context from documents using the generate_response tool. If you use the
                    generate_response tool and you are not able to answer the question because there is no provided
                    context, ask the user if it is okay to use ChatGPT's training data for getting an answer and wait
                    for a follow-up response. If the answer is yes then use the general knowledge to give the answer,
                    else do not."""),
                ("user", """If the user's query is about diagnosing a problem, providing information from the documents,
                    or answering general questions about information in the documents, use the 'generate_response' tool.
                    The user's query is: {input}"""),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        llm_with_tools = llm.bind_tools(tools)

        agent = (
                {
                    "input": lambda x: x["input"],
                    "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                        x["intermediate_steps"]
                    )
                }
                | main_prompt
                | llm_with_tools
                | OpenAIToolsAgentOutputParser()
        )

        return AgentExecutor(agent=agent, tools=tools, verbose=False)

    def chat_with_agent(self, input_text):
        #print("**********input--->", input_text)
        result = self.agent_executor(
            {"input": input_text}
        )

        response = result["output"]

        return response

# Helper function to format context for the LLM
def format_context(retrieved_context):
    """Format the context into a structure suitable for the LLM."""
    # For example, you could join the text from the retrieved documents or create a structured prompt
    return "\n".join([doc["text"] for doc in retrieved_context])

# Streamlit interface
def streamlit_interface():
    # Sidebar input
    st.sidebar.title("Upload PDF Files for Processing")
    
    # File uploader with a unique key to avoid duplicates
    uploaded_files = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf", key="file_uploader")
    
    # Debugging output to check uploaded files
    print(f"Uploaded files: {uploaded_files}")  # Check the state of uploaded_files

    if st.sidebar.button("Process PDF Files"):
        if uploaded_files:
            try:
                # Initialize vectorstore with uploaded files
                initialize_vectorstore(uploaded_files)
                uploaded_files = None
            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}")
        else:
            st.sidebar.warning("Please upload some PDF files.")

    # Main chat interface
    st.title("Document-based Chatbot")

    # Instantiate the chat interface
    chat_interface = ChatInterface()

    # User input for question

    if "chatbot" not in st.session_state:
        st.session_state["chat_interface"] = chat_interface
    
    chatbot = st.session_state["chat_interface"]
            
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I am a PDF chatbot, You can upload PDF, ask questions from it and I will answer..."}]
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        with st.spinner("Generating Response..."):
            response = chatbot.chat_with_agent(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(str(response))

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_interface()
