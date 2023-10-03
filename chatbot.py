import streamlit as st
import os
from PIL import Image
from tempfile import TemporaryDirectory
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# Initialize global variables
chat_history = []
chain = None

# Define a class for the chatbot application
class ChatbotApp:
    def __init__(self):
        self.api_key = None
        self.pdf_file = None
        self.pdf_loaded = False

    def set_api_key(self, api_key):
        os.environ['OPENAI_API_KEY'] = api_key
        st.success('OpenAI API key is set.')

    def process_pdf(self, pdf_file):
        with TemporaryDirectory() as tmp_dir:
            pdf_file_path = os.path.join(tmp_dir, pdf_file.name)
            with open(pdf_file_path, "wb") as f:
                f.write(pdf_file.read())
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()
            embeddings = OpenAIEmbeddings()
            pdfsearch = Chroma()
            pdfsearch.from_documents(documents, embeddings)
            global chain
            chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3),
                                                          retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
                                                          return_source_documents=True)
            self.pdf_loaded = True

    def add_text(self, text):
        if not text:
            st.warning('Enter text.')
        else:
            chat_history.append(text)

    def generate_response(self, query):
        if not self.pdf_loaded:
            st.warning('Upload a PDF before generating a response.')
            return
        result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
        chat_history.append(query + '\n' + result["answer"])
        st.success('Response generated successfully.')

    def render_pdf(self):
        if not self.pdf_loaded:
            st.warning('Upload a PDF first.')
            return
        file = st.file_uploader("Upload PDF", key="pdf_uploader", type=["pdf"])
        if file is not None:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".pdf") as out_pdf:
                out_pdf.write(file.getvalue())
                out_pdf.flush()
                
               # Convert the first page of the PDF to an image
                images = convert_from_path(out_pdf.name)
                if images:
                    st.image(images[0])

def main():
    st.title("Streamlit Chatbot with PDF Search")
    chatbot = ChatbotApp()

    st.sidebar.header("OpenAI API Key")
    api_key = st.sidebar.text_input("Enter OpenAI API key")
    if st.sidebar.button("Set API Key"):
        chatbot.set_api_key(api_key)

    st.sidebar.header("PDF Upload")
    pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file is not None:
        chatbot.process_pdf(pdf_file)

    st.header("Chatbot")
    text = st.text_area("Enter text and press enter:")
    if st.button("Submit"):
        chatbot.add_text(text)

    query = st.text_input("Enter a query for the chatbot:")
    if st.button("Generate Response"):
        chatbot.generate_response(query)

    st.header("PDF Viewer")
    chatbot.render_pdf()

if __name__ == "__main__":
    main()
