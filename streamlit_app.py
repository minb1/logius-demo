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
st.title("Logius Docs")

# Sidebar for user input
st.sidebar.header("Query Settings")
top_k = st.sidebar.slider("N Docs retrieved", 1, 50, 10)
user_query = st.sidebar.text_area("Prompt:")

if st.sidebar.button("Submit Query") and user_query:
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
    
    # Read and clean retrieved chunk texts
    retrieved_texts = []
    markdown_files = set()
    
    for relative_path in chunk_paths:
        if not relative_path:
            continue
        
        normalized_path = relative_path.replace("\\", "/")
        full_file_path = os.path.join(data_folder, "chunks", normalized_path)
        
        try:
            with open(full_file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
            retrieved_texts.append(content)
            
            # Extract markdown file name from chunk path
            markdown_file = os.path.dirname(normalized_path).split("/")[-1] + ".md"
            markdown_files.add(markdown_file)
        except FileNotFoundError:
            st.error(f"File not found: {full_file_path}")
        except Exception as e:
            st.error(f"Error reading file '{full_file_path}': {e}")
    
    context_text = "\n\n".join(retrieved_texts)
    
    # Query Gemini model
    prompt = f"""
    Beantwoord de volgende vraag op basis van de verstrekte context.
    Geef een zo volledig en nauwkeurig mogelijk antwoord.
    
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
        st.subheader("💬 Chat")
        st.text_area("User Query:", value=user_query, height=100, disabled=True)
        st.subheader("🤖 Chatbot Response")
        st.write(chatbot_response)
    
    # Middle: Retrieved document chunks
    with col2:
        st.subheader("📄 Answer is based on this context:")
        for i, text in enumerate(retrieved_texts):
            with st.expander(f"Chunk {i+1}"):
                st.write(text)
    
    # Right: Full markdown files
    with col3:
        st.subheader("📜 Full Markdown Files")
        for md_file in markdown_files:
            md_path = os.path.join(data_folder, "markdown_files", md_file)
            try:
                with open(md_path, "r", encoding="utf-8") as file:
                    md_content = file.read()
                with st.expander(md_file):
                    st.markdown(md_content)
            except FileNotFoundError:
                st.error(f"Markdown file not found: {md_file}")
            except Exception as e:
                st.error(f"Error reading markdown file '{md_file}': {e}")
