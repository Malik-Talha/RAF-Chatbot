import streamlit as st
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_community.tools.render import format_tool_to_openai_function
from streamlit.runtime.uploaded_file_manager import UploadedFile
import tempfile
from typing import List
import zipfile
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.qdrant import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain.agents.agent import AgentExecutor

from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages.human import HumanMessage

def main():
    # set basic layout of bot
    st.set_page_config('Chatbot')
    st.header('Chat')

    # create session variables
    if 'processComplete' not in st.session_state:
        st.session_state.processComplete = None

    # set sidebar for file uploads
    with st.sidebar:
        files_uploaded = st.file_uploader('Upload Books', type=['pdf', 'docx', 'zip'], accept_multiple_files=True)
        process = st.button('Process')

    if process:
        if len(files_uploaded) == 0:
            st.warning('Please upload documents files to process.', icon='⚠')
        else:
            # get documents from uploaded files
            documents: List[Document] = get_documents(files_uploaded)
            # split documents using 'Split by character'
            # chunks: List[Document] = simple_split_documents(documents)

            # split documents using 'Recursively split by character' (recommended)
            chunks = recursively_split_documents(documents)

            # # create vector embeddings of documents using BAAI/bge-small-en-v1.5
            # embeddings_model: Embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

            # create vector embeddings of documents using intfloat/e5-large-v2 (recommended)
            embeddings_model: Embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

            # create vectorstore (store-backed retriever)
            vectorstore: Qdrant = get_qdrant_vectorstore(chunks, embeddings_model)

            # get agent_executor runnable
            agent_executor: AgentExecutor = get_agent_executor(vectorstore)

            # keep vectorstore in session state if new
            st.session_state.agent_executor = agent_executor

            # save the progress in session
            st.session_state.processComplete = True

    # let user query after successful process completion
    if st.session_state.processComplete == True:
        handle_user_queries()


@st.cache_data(show_spinner='Loading Files...')
def get_documents(files: List[UploadedFile]) -> List[Document]:
    """
    returns documents from uploaded files (PDFs, Docs)
    :param files: list of UploadedFile
    :return: List of documents from the document files
    """

    documents = []
    for file in files:
        # create a temporary file as path feature is not available in the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name  # Get temporary file path
            # Use temp_file_path for operations like os.stat(), the tempfile will be deleted when garbage collector run

        # load if file is pdf
        if file.type == 'application/pdf':
            loader = PyMuPDFLoader(temp_file_path)
            documents.extend(loader.load())


        # load if file is docx
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            loader = Docx2txtLoader(temp_file_path)
            documents.extend(loader.load())

        # if zip file of type: 'application / x - zip - compressed'
        else:
            with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                # Extract to a temporary directory
                zip_ref.extractall(tempfile.gettempdir())

            # Access PDF and DOCX files within the extracted directory
            extracted_dir = tempfile.gettempdir()
            for filename in os.listdir(extracted_dir):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(extracted_dir, filename)
                    # load
                    loader = PyMuPDFLoader(temp_file_path)
                    documents.extend(loader.load())
                elif filename.endswith(".docx"):
                    file_path = os.path.join(extracted_dir, filename)
                    # load docx
                    loader = Docx2txtLoader(temp_file_path)
                    documents.extend(loader.load())
            # Clean up temporary files
            os.remove(temp_file_path)  # Delete the temporary ZIP file
            for filename in os.listdir(extracted_dir):
                os.remove(os.path.join(extracted_dir, filename))  # Delete extracted files

    return documents


def simple_split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks and return list of documents
    :param documents: list of documents to split
    :return: List of chunks
    """
    # define splitter
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    # show progress to user
    with st.spinner(text="creating chunks...", cache=False):
        # split documents into chunks
        chunks = splitter.split_documents(documents)
    return chunks


def recursively_split_documents(documents: List[Document]) -> List[Document]:
    """
     Recursively split documents into chunks and return list of documents
    :param documents: list of documents to split
    :return: List of chunks
    """
    # define splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    # split documents into chunks
    chunks = splitter.split_documents(documents)
    return chunks


def get_qdrant_vectorstore(docs: List[Document], embeddings_model: Embeddings) -> Qdrant:
    """
     returns Qdrant vectorstore for the given documents and embeddings
    :param docs: chunks of documents
    :param embeddings_model: embeddings model to be used to create vectorDB
    :return: Qdrant vectorstore
    """

    # show message and spinner to the user while embeddings are being created
    with st.spinner(text='Creating Vectorstore... This may take a few minutes'):
        qdrant = Qdrant.from_documents(
            docs,
            embeddings_model,
            url=st.secrets['qdrant_url'],
            prefer_grpc=True,
            api_key=st.secrets['qdrant_api'],
            collection_name="my_documents",
            force_recreate=False,
        )
    return qdrant


def get_agent_executor(vectorstore: Qdrant) -> AgentExecutor:
    """
    Returns AgentExecutor runnable
    :param vectorstore: vectorstore created with embeddings and documents
    :return: AgentExecutor
    """
    # create chat LLM
    chat_llm = ChatOpenAI(model="gpt-3.5-turbo-16k", openai_api_key=st.secrets['openai_api'], temperature=0.0)

    tool = create_retriever_tool(
        vectorstore.as_retriever(),
        "get_context",
        "Searches and returns context from the documents to answer each of user queries. Pass the user query as it is to it and it will return useful contexts",
    )
    tools = [tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant who answers each of user query by 'get_context' function available to you. You answer all the questions by always using the document retrieval provided to you. You never answer from your own knowlege or from anywhere else except the context from the tool.\n"),
            MessagesPlaceholder(variable_name='chat_history'),
            ("human","{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)



    # Bind and format tools as OpenAI Functions
    llm_with_tools = chat_llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools]
    )

    # Build the agent. Give it a scratchpad and history.
    agent = {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps']),
                "chat_history": lambda x: x["chat_history"]
            } | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()

    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
    
    return agent_executor


def display_messages(chat_history: List) -> None:
    """
    display all the history and new messages of Human and AI to user
    :param chat_history: history or list of human and ai messages
    :return: none
    """

    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("human"):
                st.write(msg.content)
        else:
            with st.chat_message("ai"):
                st.write(msg.content)


def handle_user_queries():
    """ handle user queries """
    # get user query
    query = st.chat_input("Ask something relevant to the documents")

    # if user sends message
    if query:
        # get AgentExecutor from the session
        agent_executor: AgentExecutor = st.session_state.agent_executor
        res = agent_executor.invoke({'input': query})

        # get history with current message and response
        chat_history = res['chat_history']

        # display messages
        display_messages(chat_history)




if __name__ == '__main__':
    main()
