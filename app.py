import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Define directories
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Initialize embedding function
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Check if vector store exists
if not os.path.exists(persistent_directory):
    st.error(f"Vector store not found at {persistent_directory}. Please run the document indexing script first.")
    st.stop()

# Load vector store
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Set up retriever with MMR search
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 7})

# Initialize the language model
model = ChatOpenAI(model="gpt-4o", temperature=0.5)

# Define the system message
system_message_content = """You are an AI assistant functioning as a knowledgeable and objective guide to the Constitution of India and related Indian laws. Your core mission is to help users understand their fundamental rights and basic legal principles as outlined **strictly and solely** within the provided document excerpts. You aim to make this knowledge accessible in **extremely simple and clear language**, empowering users to be aware of their rights and the basic legal landscape.

**Core Directives:**

1.  **Absolute Textual Grounding:** Your answers MUST be based *exclusively* on the information present in the 'Relevant Constitution of India documents' provided in the prompt context. Do NOT add external knowledge, interpretations, case laws, or personal opinions. This is crucial for accuracy and reliability.
2.  **Mandatory & Precise Citations:** For *every* piece of information, legal principle, or explanation you provide, you MUST cite the source document and page number (if available) from the context. Use the format: `(Source: [Filename], Page: [PageNumber])` or `(Source: [Filename])` if page is unavailable. If multiple sources support a point, cite them all relevantly. Accuracy in citation is paramount.
3.  **Extreme Simplicity & Clarity:** Explain complex legal and constitutional concepts using everyday language. Avoid jargon. If a legal term (like an Article number or a specific Act section) is necessary, *immediately explain its meaning or relevance* in simple terms. Pretend you are explaining to someone completely unfamiliar with the Indian legal system.
4.  **Handling Situational Questions (Crucial Safety Guideline):**
    *   Users might describe a situation and ask if it's "legal," "right," or what their rights are.
    *   **DO NOT:** Give direct legal advice, judge the legality of their specific, personal situation, predict outcomes, or tell them what they *should* do.
    *   **DO:**
        *   Identify the general legal principles, Constitutional Articles, or law sections *from the provided documents* that seem relevant to the *type* of situation described.
        *   Explain *what the law (as presented in the documents) says* about those principles or rules in general.
        *   Clearly state that you are explaining the general legal position based *only* on the provided text and cannot comment on their specific case.
        *   Use cautious framing: "According to the provided text from Article [X] (Source:...), the Constitution guarantees [...]. Section [Y] of the [Act Name] (Source:...) outlines rules regarding [...]. Generally, based on these documents, situations like the one you described might involve considerations of [...], but I cannot assess your specific circumstances."
5.  **Acknowledge Missing Information:** If the provided documents do not contain information relevant to the user's question, or if the context is insufficient to form a reliable explanation, state clearly and directly: "Based *only* on the documents provided, I cannot find specific information about [topic of the question] or cannot provide a sufficiently detailed explanation." Do not guess or try to fill gaps.
6.  **Prioritize the Constitution:** When discussing topics covered by the Constitution, explain the constitutional provisions first. Use supplementary documents (like commentaries or other Acts provided in the context) to elaborate or provide related procedures, always citing them clearly.
7.  **Neutral, Objective, and Empowering Tone:** Maintain a factual, unbiased tone. Your aim is to inform and educate neutrally, thereby empowering the user with accurate knowledge derived *only* from the provided texts. Avoid alarmist or overly strong language.
8.  **Explaination in Simple term**  : If asked to explain in simple terms, use simple language and avoid jargon.
**Final Check before Responding:** Does my answer strictly adhere to the provided text? Is every key point cited? Is the language simple enough? Have I avoided giving specific legal advice?
"""
system_message = SystemMessage(content=system_message_content)

# Define global constants
MAX_HISTORY_PAIRS = 3

# Define Streamlit page configuration
st.set_page_config(
    page_title="Constitution of India Expert",
    page_icon="📜",
    layout="wide"
)

# Define function to get streaming response
def get_streaming_response(user_message):
    # Retrieve relevant documents
    with st.spinner("Searching for relevant information..."):
        relevant_docs = retriever.invoke(user_message)
    
    # Format context with citations
    context_parts = []
    if relevant_docs:
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get('source', 'Unknown Source')
            page_num = doc.metadata.get('page', None)
            source_cite = f"Source: {source}" + (f", Page: {page_num + 1}" if page_num is not None else "")
            context_parts.append(f"--- Context Document {i + 1} ---\n{source_cite}\nContent Snippet: {doc.page_content}\n--- End Context Document {i + 1} ---")
    else:
        context_parts.append("--- No relevant documents found in knowledge base. Try asking about a specific constitutional article or law for a more detailed response! ---")
    
    context = "\n\n".join(context_parts)
    
    # Summarize conversation history
    if "history" not in st.session_state:
        st.session_state.history = []
    
    history_summary = "Conversation so far:\n"
    if st.session_state.history:
        for h_msg, a_msg in st.session_state.history[-MAX_HISTORY_PAIRS:]:
            history_summary += f"- User asked: {h_msg}\n  Assistant replied: {a_msg[:100]}...\n"
    else:
        history_summary += "No prior conversation.\n"
    
    # Create enhanced user message with context
    enhanced_user_message_content = f"""{history_summary}
Here is the user's latest question:
{user_message}

--- Relevant Constitution of India documents ---
{context}
--- End Relevant Documents ---

Based *only* on the provided documents above, please answer the user's question following all instructions in the initial System Message (especially regarding citations, simple language, and not giving legal advice). Provide a detailed explanation of any relevant general principles or rights from the documents, even if they don’t directly address the specific situation. If the context is empty or irrelevant, state that you cannot answer fully and suggest how the user might refine their question (e.g., by asking about a specific article or law)."""
    
    # Handle ambiguity
    if "it" in user_message.lower() and st.session_state.history:
        enhanced_user_message_content += "\nNote: I assume 'it' refers to the previous topic. If not, please clarify what you mean by 'it'."
    
    # Build message list for the LLM
    messages = [system_message]
    history_pairs = st.session_state.history[-MAX_HISTORY_PAIRS:] if st.session_state.history else []
    for h_msg, a_msg in history_pairs:
        messages.append(HumanMessage(content=h_msg))
        messages.append(AIMessage(content=a_msg))
    messages.append(HumanMessage(content=enhanced_user_message_content))
    
    # Get streaming response
    with st.spinner("Generating response..."):
        response = model.stream(messages)
    return response, context  # Return context for UI display

# Main app function
def main():
    # Add title to sidebar
    with st.sidebar:
        st.title("🇮🇳 Constitution of India Expert")
        st.markdown("---")
        st.markdown("""
        This assistant helps you understand the Constitution of India and related laws 
        based on the documents in its knowledge base.
        
        Ask any question about your fundamental rights, legal principles, or constitutional provisions.
        """)
        st.markdown("---")
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Display chat header
    st.header("Constitution of India Assistant")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    user_query = st.chat_input("Ask about the Constitution of India...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            response_stream, context = get_streaming_response(user_query)
            for chunk in response_stream:
                if chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            with st.expander("View Sources"):
                st.markdown(context.replace("\n", "\n\n"))
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Store the conversation pair in history for future context
        st.session_state.history.append((user_query, full_response))

if __name__ == "__main__":
    main()