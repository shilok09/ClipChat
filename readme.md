# ClipChat

**ClipChat** is a Retrieval-Augmented Generation (RAG) conversational chatbot that allows you to upload an audio clip, automatically transcribes it, stores the transcript in a vector database, and then answers your questions about the content of the audio using OpenAI's GPT-4.1 model. The bot only answers using information retrieved from the transcript, and will say "I don't know" if the answer is not present in the context.

---

## Features

- **Audio-to-Text**: Upload an audio file and get an automatic transcript using the Gladia API.
- **RAG Workflow**: Stores transcript chunks in a FAISS vector database for efficient retrieval.
- **Conversational QA**: Ask questions about the audio/video content in a chat loop.
- **Context-Restricted Answers**: The AI only answers from the retrieved transcript context.
- **LangGraph Workflow**: Modular, node-based workflow using LangGraph for flexible orchestration.
- **OpenAI GPT-4.1**: Uses OpenAI's latest GPT-4.1 model for chat responses.

---

## Setup

### 1. Clone the repository

```sh
git clone https://github.com/shilok09/ClipChat.git
cd ClipChat
```

### 2. Create and activate a virtual environment

```sh
python -m venv venv
venv\Scripts\activate   # On Windows
# Or
source venv/bin/activate  # On Mac/Linux
```

### 3. Install dependencies

```sh
pip install -r requirements.txt
```

### 4. Set up your `.env` file

Create a `.env` file in the root directory with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1
GLADIA_API_KEY=your_gladia_api_key
GLADIA_API_URL=https://api.gladia.io/audio/text/audio-transcription/
LANGSMITH_API_KEY=your_langsmith_api_key
```

**Never commit your `.env` file!**

---

## Usage

1. Place your audio file (e.g., `short-audio.wav`) in a known directory.
2. Update the `audio_path` in the script if needed.
3. Run the chatbot:

```sh
python agent.py
```

4. Interact with the bot in your terminal:

```
YOU: What is this audio about?
AI: [Answer from transcript context]
```

Type `exit` to quit.

---

## Project Structure

```
ClipChat/
│
├── agent.py           # Main chatbot workflow
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignores .env and venv/
└── README.md          # Project documentation
```

---

## Notes

- The bot will only answer using the transcript context. If the answer is not found, it will respond with "I don't know."
- The first run with an audio file will transcribe and index the transcript. Subsequent questions will use the stored transcript.
- Make sure your API keys are valid and have sufficient quota.

---

## License

MIT License

---

## Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph)
-