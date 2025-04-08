import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# --- Embedding Function ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Load Vector Store ---
print("Loading vector store...")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
print("Vector store loaded.")

# --- Retriever ---
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 7}, # Retrieve top 5 most similar chunks
)
print("Retriever created.")

# --- LLM ---
model = ChatOpenAI(
    model="gpt-4o", # Or gpt-3.5-turbo for faster/cheaper responses
    temperature=0.5 # Lower temperature for more factual, less creative answers
)
print("ChatOpenAI model initialized.")

# --- System Prompt ---
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

**Final Check before Responding:** Does my answer strictly adhere to the provided text? Is every key point cited? Is the language simple enough? Have I avoided giving specific legal advice?
"""
system_message = SystemMessage(content=system_message_content)

# Initialize conversation history
conversation_history = [system_message]

print("\nConstitution of India Expert Assistant (Type 'exit' to quit)")
print("---------------------------------------------------------")

while True:
    # Get user input
    user_message_content = input("\nYou: ")

    # Check for exit command
    if user_message_content.lower() in ["exit", "quit", "bye", "goodbye"]:
        print("\nAI: Thank you for using the Constitution of India Expert Assistant. Goodbye!")
        break

    # Add the simple user message to history (for context in multi-turn)
    conversation_history.append(HumanMessage(content=user_message_content))

    try:
        print("AI: Searching relevant documents...")
        # Retrieve relevant documents based on the latest user query
        relevant_docs = retriever.invoke(user_message_content)

        # Format retrieved context
        context_parts = []
        if relevant_docs:
             for i, doc in enumerate(relevant_docs):
                source = doc.metadata.get('source', 'Unknown Source')
                # Include page number if available in metadata (PyPDFLoader often adds it)
                page_num = doc.metadata.get('page', None)
                source_cite = f"Source: {source}" + (f", Page: {page_num+1}" if page_num is not None else "") # PyPDFLoader pages are 0-indexed
                context_parts.append(f"--- Context Document {i+1} ---\n{source_cite}\nContent Snippet: {doc.page_content}\n--- End Context Document {i+1} ---")
        else:
            context_parts.append("--- No relevant documents found in the knowledge base. ---")

        context = "\n\n".join(context_parts)

        # Create the enhanced prompt for the LLM, including history and context
        # We rebuild the message list for the LLM each time to include the latest context
        llm_input_messages = conversation_history[:-1] # History *before* the last user message

        enhanced_user_message_content = f"""Here is the user's latest question:
{user_message_content}

--- Relevant Constitution of India documents ---
{context}
--- End Relevant Documents ---

Based *only* on the provided documents above, please answer the user's question following all instructions in the initial System Message (especially regarding citations, simple language, and not giving legal advice). If the context is empty or irrelevant, state that you cannot answer from the provided documents."""

        # Add the enhanced user message to the list for the LLM
        llm_input_messages.append(HumanMessage(content=enhanced_user_message_content))

        print("AI: Thinking...")
        # Inside your loop, instead of response = model.invoke(...)
        response_stream = model.stream(llm_input_messages)
        print("\nAI: ", end="")
        full_response_content = ""
        for chunk in response_stream:
            content = chunk.content
            if content:
                print(content, end="", flush=True)
                full_response_content += content
        print() # Newline after streaming finishes

        # Add the complete message to history
        conversation_history.append(AIMessage(content=full_response_content))
        # Optional: Limit history size to prevent excessive token usage
        # For example, keep only the system message and the last N interactions
        MAX_HISTORY_TURNS = 5
        if len(conversation_history) > (1 + 2 * MAX_HISTORY_TURNS): # 1 system + 2 per turn (Human/AI)
            conversation_history = [system_message] + conversation_history[-(2 * MAX_HISTORY_TURNS):]


    except Exception as e:
        print(f"\nAI: I apologize, but I encountered an error: {str(e)}")
        # Optionally remove the last human message from history if call failed
        conversation_history.pop()