# use to upload single file on streamlit 
# Streamlit UI

# Import necessary libraries
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from utils import create_vector_db, load_model_and_tokenizer, streamlit_parse_file 
import configs
import torch 

# Cached function to load model and tokenizer
@st.cache_resource()
def cached_load_model(model_name_or_path): 
    return load_model_and_tokenizer(model_name_or_path) 

# ...

loaded_model = cached_load_model(configs.model_name_or_path) 
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device': 'cpu'}) # Ensure this is 'cpu'
db = FAISS.load_local(configs.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True) 
print("Model, tokenizer, and db loaded successfully.")

# Streamlit app title
st.title("Metadata Generation")

# File upload section
uploaded_file = st.file_uploader("Upload a file", type=[".pptx", ".docx", ".xlsx", ".xls", ".csv", ".ipynb", ".py", ".md", ".pdf"])

if uploaded_file is not None:
    st.success("File successfully uploaded!")
    try:
        texts= streamlit_parse_file(uploaded_file)
        if texts:
            print("files parsed successfully")
            st.write("File parsed successfully!") # User feedback
        else:
            print("No file parsed")
            st.warning("No text extracted from the file.")
            # exit() # Don't exit Streamlit app like this

        print("\nCreate vector db...")
        try:
            create_vector_db(texts)
            st.success("Vector database created successfully for the uploaded file.")
        except Exception as e:
            print(e)
            st.error(f"Error creating vector database: {e}")
            # exit() # Don't exit Streamlit app like this

        print("Vector db created successfully.")        
        
        
        # The Streamlit app will now simply confirm successful parsing and DB creation.

    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
    

