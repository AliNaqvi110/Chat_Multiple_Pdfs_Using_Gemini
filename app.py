# import libraries
import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from htmlTemplates import css, bot_template, user_template


# funvtion to read pdf document
def get_pdf_text(docs):
    text = ""
    for doc in docs:         # for each document
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:      # for each page
            text += page.extract_text()    # extract text
    return text

# function to make chunks of a text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# get vectorstore
def get_vectorstore(text_chunks):
    # generate embddings
    embeddings =  GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# get conversations
def get_conversations(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# handle user question
def  handleQuestion(user_question):
    if st.session_state.conversations is not None:  # Check if conversations is not None
        response = st.session_state.conversations({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)





# main funvtion
def main():
    # load environment variables
    try:
        load_dotenv()
    except:
        # Raise an exception
        raise RuntimeError("GEMINI_API_KEY environment variable is required!")
    
    # set layout
    st.set_page_config(page_title="Pdf Chatbot", page_icon=":books")

    # Add css
    st.write(css, unsafe_allow_html=True)

    # initialize session state
    if "conversations" not in st.session_state:
        st.session_state.conversations=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None

    # add header
    st.header("Gemini Pro Pdf Chatbot :books")
    user_question = st.text_input("Ask a question about your document")

    if user_question:
        handleQuestion(user_question)



    # side bar
    with st.sidebar:
        st.subheader("Your Document")                                                   # Sub Header
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)     # file Uploader 
        if st.button("Process"):                                                        # button
            
            # Add Spinner
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)



                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversations
                st.session_state.conversations = get_conversations(vectorstore)

    

# calling main function
if __name__ == '__main__':
    main()