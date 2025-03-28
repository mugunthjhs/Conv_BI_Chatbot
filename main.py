import streamlit as st
import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from preprocess import process_sales_data

def load_data():
    @st.cache_data
    def inner_load_data():
        if os.path.exists("sales_data.xlsx"):
            return pd.read_excel("sales_data.xlsx")
        else:
            st.error("Oops! It seems the sales data file is missing. Please check the file path.")
            return pd.DataFrame()
    return inner_load_data()

def get_vector_store(data, faiss_index_path):
    @st.cache_resource
    def inner_get_vector_store(data):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

        if os.path.exists(faiss_index_path):
            return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

        if not data.empty:
            contexts = process_sales_data(data)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = [chunk for context in contexts for chunk in text_splitter.split_text(context)]
            
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local(faiss_index_path)
            return vector_store
        return None

    return inner_get_vector_store(data)

def process_user_question(user_question, faiss_index_path):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    new_db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(google_api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']

def get_conversational_chain(api_key):
    @st.cache_resource
    def inner_get_conversational_chain(api_key):
        prompt_template ="""
                As a **business intelligence expert** in the pharmaceutical supply chain, please provide **detailed and insightful answers** to the following questions. Ensure that your responses demonstrate a **thorough understanding** of **data analysis, visualization, and business strategy**. Support your answers with **relevant data or examples** where appropriate.

                Provide a **concise and accurate answer** to the question posed.
                Include **relevant metrics and data points** to support your answer.

                **Insights:**
                - **Summarize significant trends** observed in the data.
                - **Highlight any notable changes or patterns** over time.

                **Data Table:**
                - Present a **clear table of relevant sales figures or data points** that directly relate to the question. If the user requests the top 10 items, provide 10 rows; if they request the top 5, provide 5 rows, and so on, based on their specific request.

                **Business Impact (if applicable):**
                - Discuss the **implications of these trends on the business**.
                - **Compare findings with historical data** where relevant.

                **Recommendations:**
                - Offer **actionable recommendations** based on the analysis.

                **Illustrations are not necessary.**

                **Context:**\n{context}\n
                **Question:** \n{question}\n
            **Answer:**
            """

        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return inner_get_conversational_chain(api_key)

def main():
    # Load environment variables
    load_dotenv()
    FAISS_INDEX_PATH = "faiss_index"

    # Streamlit UI
    st.set_page_config(page_title="Conversational BI Chatbot", page_icon="📊", layout="wide")

    # Center title and subheading
    st.markdown("<h1 style='text-align: center;'>Conversational Business Intelligence Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Ask your questions about the  Pharmaceutical Supply Chain Data below:</p>", unsafe_allow_html=True)


    # Change theme to white
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: white;
            color: black;
        }
        .sidebar .sidebar-content {
            background-color: #f0f0f0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    sales_data = load_data()

    vector_store = get_vector_store(sales_data, FAISS_INDEX_PATH)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("What would you like to know about the sales data?", key="left_align"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        response_output = process_user_question(user_question, FAISS_INDEX_PATH)
        
        with st.chat_message("assistant"):
            st.markdown(response_output)
        
        st.session_state.messages.append({"role": "assistant", "content": response_output})
        
if __name__ == "__main__":
    main()