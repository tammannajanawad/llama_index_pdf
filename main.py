import streamlit as st
from llama_index.core import PromptTemplate, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import os
import tempfile
from dotenv import load_dotenv
load_dotenv()
llm = OpenAI(model="gpt-4o-mini")
embedding_model = OpenAIEmbedding(model="text-embedding-3-small")

TEXT_QA_SYSTEM_TMPL = (
    "You answer questions using ONLY the following excerpts from the user's "
    "uploaded PDFs. Do not use outside knowledge, the web, or any facts not "
    "stated in these excerpts. If the excerpts do not contain enough "
    "information, say the documents do not state that and avoid guessing.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)
text_qa_template = PromptTemplate(
    TEXT_QA_SYSTEM_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)
REFINE_SYSTEM_TMPL = (
    "The user's PDF text may be split across several passages. Refine the "
    "answer using ONLY the new passage below, together with the prior "
    "excerpts already considered. Do not add knowledge from outside those "
    "excerpts. If the new passage is not useful, return the previous answer.\n"
    "The original query is as follows: {query_str}\n"
    "The existing answer is: {existing_answer}\n"
    "New passage:\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "If the new passage is relevant, refine the answer. Otherwise return the existing answer.\n"
    "Refined answer: "
)
refine_template = PromptTemplate(REFINE_SYSTEM_TMPL, prompt_type=PromptType.REFINE)
# -----------------------------------
# Streamlit UI for document uploading
# -----------------------------------
st.title("Q&A with Your Documents")

st.markdown("Upload your `.pdf` files to build a semantic search index and ask questions.")

uploaded_files = st.file_uploader("Upload text files", type="pdf", accept_multiple_files=True)

query = st.text_input("Ask a question about the documents")
if uploaded_files and query:
    with st.spinner("Processing documents and building index..."):
        # Save uploaded files to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Load documents
            documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
            
            # Create vector index
            index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
            # Retrieve and query
            query_engine = index.as_query_engine(
                llm=llm,
                text_qa_template=text_qa_template,
                refine_template=refine_template,
            )
            response = query_engine.query(query)

            # Display results
            st.subheader("Answer:")
            st.write(response.response)
