# Ask_Truth
This is a chatbot that I made using Langchain and RAG 

# Constitution of India Expert Assistant

A Streamlit-based AI assistant that helps users understand the Indian Constitution and related laws using provided document excerpts and precise source citations.

## Features
- **Document Retrieval:** Uses a Chroma vector store for fast search of constitutional texts.
- **LLM-Powered Responses:** Answers strictly based on provided documents with clear citations.
- **Streaming Responses:** Displays answers as they are generated.
- **Conversation Context:** Maintains recent conversation history for context.

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/constitution-india-expert.git
   cd constitution-india-expert
   ```

2. **Set Up Environment:**
   - Create a virtual environment and activate it.
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Create a `.env` file with your OpenAI API key:
     ```env
     OPENAI_API_KEY=your_openai_api_key_here
     ```

3. **Prepare Data:**
   - Run the document indexing script to generate the Chroma vector store under `db/chroma_db_with_metadata`.

## Usage
Run the app with:
```bash
streamlit run app.py
```


## License & Disclaimer
Released under the MIT License. This assistant is for educational purposes only and does not provide personalized legal advice.
```

This shorter README covers the essential details while ensuring clear guidance for setup and usage.
