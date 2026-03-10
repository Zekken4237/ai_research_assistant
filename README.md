# ai_research_assistant

Research assistant project.

## Setup

This app now uses a local Ollama model by default.

Make sure Ollama is running on `http://127.0.0.1:11434` and that `llama2-uncensored:7b` is available.

You can optionally create a `.env` file in the project root or `backend/.env` with:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama2-uncensored:7b
```

Restart the backend after changing the model settings.
