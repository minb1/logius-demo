import os
import streamlit as st
from google import genai
from pinecone import Pinecone

data_folder = "data"  # Base folder containing chunks and markdown_files

# Initialize Google and Pinecone clients
google_client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("google-embed-004-768d")

# Streamlit UI
st.set_page_config(layout="wide")

# Sidebar for user input
st.sidebar.header("Query Settings")
top_k = st.sidebar.slider("N aantal documenten te gebruiken", 1, 100, 50)
user_query = st.sidebar.text_area("Prompt:")

if st.sidebar.button("Stuur vraag ->") and user_query:
    # Get embeddings
    embed_result = google_client.models.embed_content(
        model="text-embedding-004",
        contents=[user_query]
    )
    query_embedding = embed_result.embeddings[0].values

    # Query Pinecone
    response = index.query(
        namespace="ns1",
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )

    # Extract file paths
    chunk_paths = [match["metadata"].get("file_path") for match in response["matches"]]

    # Read and clean retrieved chunk texts and collect markdown files
    retrieved_texts = []
    markdown_files = set()

    for relative_path in chunk_paths:
        if not relative_path:
            continue

        normalized_path = relative_path.replace("\\", "/")
        full_chunk_path = os.path.join(data_folder, "chunks", normalized_path)

        try:
            with open(full_chunk_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
            retrieved_texts.append(content)

            # Determine the folder for markdown files based on the chunk's parent directory.
            parent_dir = os.path.dirname(normalized_path)  # e.g., "Logius-standaarden_ADR-Beheermodel/Abstract_v1_0"
            md_dir = os.path.join(data_folder, "markdown_files", parent_dir)

            # Collect all markdown files in the corresponding directory.
            if os.path.exists(md_dir) and os.path.isdir(md_dir):
                for filename in os.listdir(md_dir):
                    if filename.endswith(".md"):
                        md_path = os.path.join(md_dir, filename)
                        markdown_files.add(md_path)

        except FileNotFoundError:
            st.error(f"Chunk file not found: {full_chunk_path}")
        except Exception as e:
            st.error(f"Error reading chunk file '{full_chunk_path}': {e}")

    context_text = "\n\n".join(retrieved_texts)

    # Query Gemini model
    prompt = f"""
Beantwoord de volgende vraag op basis van de verstrekte context.
Geef een zo volledig en nauwkeurig mogelijk antwoord. 
Let op technische details.

**Vraag:** {user_query}

**Context:** {context_text}

Antwoord in het Nederlands.
"""
    response = google_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    chatbot_response = response.text if response.text else "Geen antwoord beschikbaar."

    # Layout for three columns
    col1, col2, col3 = st.columns(3)

    # Left: Chatbox and response
    with col1:
        st.subheader("Chat")
        st.text_area("Jouw vraag:", value=user_query, height=100, disabled=True)
        st.subheader("Logius antwoord:")
        st.write(chatbot_response)

    # Middle: Retrieved document chunks
    with col2:
        st.subheader("Antwoord is gebaseerd on de volgende context:")
        for i, text in enumerate(retrieved_texts):
            with st.expander(f"Chunk {i+1}"):
                st.write(text)

    # Right: Full markdown files
    with col3:
        st.subheader("MD files:")
        for md_file in markdown_files:
            try:
                with open(md_file, "r", encoding="utf-8") as file:
                    md_content = file.read()
                # Use only the base filename in the expander title.
                with st.expander(os.path.basename(md_file)):
                    st.markdown(md_content)
            except FileNotFoundError:
                st.error(f"Markdown file not found: {md_file}")
            except Exception as e:
                st.error(f"Error reading markdown file '{md_file}': {e}")
