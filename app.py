import streamlit as st
from PyPDF2 import PdfReader
import tempfile
from gtts import gTTS
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_conversational_chain():
    try:
        prompt_template = """
        **Context:** {context}\n
        **Question:** {question}\n
        **Provide a comprehensive and contextually accurate response.** If the context does not contain the answer, indicate that the answer is unavailable. **Do not guess or provide potentially incorrect information.**\n
        **Answer:**
        """
        model = ChatGroq(temperature=0.5, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-13b-16384")
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        logging.error("Failed to load conversational chain: %s", str(e))
        chain = None
    return chain

def get_vector_store(text_chunks, api_key):
    try:
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        logging.info("Vector store created and saved successfully.")
    except Exception as e:
        logging.error("Failed to create vector store: %s", str(e))
        raise

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=True)
        audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_filename = audio_file.name
        tts.save(temp_filename)
        st.audio(temp_filename, format='audio/mp3')
        os.remove(temp_filename)
        logging.info("Text to speech conversion successful.")
    except Exception as e:
        logging.error("Failed to convert text to speech: %s", str(e))
        raise

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        chunks = text_splitter.split_text(text)
        logging.debug("Text split into chunks successfully.")
    except Exception as e:
        logging.error("Failed to split text: %s", str(e))
        chunks = []
    return chunks

def user_input(user_question, api_key):
    try:
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name="sentence-transformers/all-mpnet-base-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Replies:")
        if isinstance(response["output_text"], str):
            response_list = [response["output_text"]]
        else:
            response_list = response["output_text"]
        for text in response_list:
            st.write(text)
            text_to_speech(text)
    except Exception as e:
        logging.error("Failed to handle user input: %s", str(e))
        st.error("An error occurred while processing your request.")

def process_pdfs(pdf_docs):
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(extract_pdf_text, pdf_docs))
    combined_text = " ".join(results)
    logging.info("PDFs processed successfully.")
    return combined_text

def extract_pdf_text(pdf):
    text = ""
    try:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or " "
            text += page_text
        logging.debug(f"Extracted text from PDF: {pdf}")
    except Exception as e:
        logging.error("Error reading PDF file: %s", str(e))
    return text

