from dotenv import load_dotenv
import os
import csv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
import google.generativeai as genai

# Set page configuration as the very first Streamlit command
st.set_page_config(page_title="RPA Transcript Analyst", layout="wide")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Create the directory if it doesn't exist
    os.makedirs("faiss_index", exist_ok=True)
    vector_store.save_local("faiss_index")

def save_to_csv(extracted_details):
    csv_file = "transcript_details.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Date", "Time", "Customer ID", "Name", "Contact", "Email",
            "Case ID", "Issue Type", "Priority", "Agent Assigned",
            "Status", "Remarks"
        ])
        
        data = extracted_details.split("\n")
        for line in data:
            if line.strip():  # Only process non-empty lines
                writer.writerow(line.split(","))
    
    return csv_file

def get_conversational_chain():
    prompt_template = """
    Extract the following details from the provided transcript:

    1. Date of the call.
    2. Time of the call.
    3. Customer ID.
    4. Customer Name.
    5. Contact Number.
    6. Email Address.
    7. Case ID.
    8. Issue Type (e.g., Billing, Technical, Complaint).
    9. Priority: Determine the priority based on the issue type:
       - "High" for critical or urgent issues (e.g., service outage, incorrect billing).
       - "Medium" for moderately important issues.
       - "Low" for general inquiries or minor concerns.
    10. Agent Assigned (if mentioned).
    11. Status of the case.
    12. Remarks or additional comments.

    Provide the extracted details in the following format:
    Date, Time, Customer ID, Name, Contact, Email, Case ID, Issue Type, Priority, Agent Assigned, Status, Remarks.
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_transcript():
    try:
        if not os.path.exists("faiss_index/index.faiss"):
            raise FileNotFoundError("Please process the PDF documents first before generating analysis.")
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search("Extract details from the transcript.")
        
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": "Extract details"}, return_only_outputs=True)
        extracted_details = response["output_text"]
        
        csv_file = save_to_csv(extracted_details)
        return csv_file
    except Exception as e:
        st.error(f"Error processing transcript: {str(e)}")
        return None

def main():
    st.header("RPA Transcript Analyst")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload the transcript file (PDF)", accept_multiple_files=True)
        
        # Update file_uploaded state based on pdf_docs
        if pdf_docs:
            st.session_state.file_uploaded = True
        else:
            st.session_state.file_uploaded = False
            st.session_state.processed = False
        
        process_button = st.button("Submit & Process")
        
        if process_button:
            if pdf_docs:
                with st.spinner("Processing PDF..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.session_state.processed = True
                        st.success("Processing complete! Now click 'Generate' to analyze.")
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
                        st.session_state.processed = False
            else:
                st.error("Error: Please upload the file first!")
                st.session_state.processed = False

    generate_button = st.button("Generate")
    if generate_button:
        if not st.session_state.file_uploaded:
            st.error("Error: Please upload the file first!")
        elif not st.session_state.processed:
            st.error("Error: Please process the file before generating analysis!")
        else:
            with st.spinner("Analyzing the transcript..."):
                csv_file = process_transcript()
                if csv_file:
                    st.success("Details extracted and saved successfully!")
                    
                    with open(csv_file, "rb") as file:
                        st.download_button(
                            label="Download CSV",
                            data=file,
                            file_name="transcript_details.csv",
                            mime="text/csv",
                        )

if __name__ == "__main__":
    main()