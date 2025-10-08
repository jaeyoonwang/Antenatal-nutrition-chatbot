# Antenatal Nutrition Chatbot

Streamlit application that lets you demo an antenatal nutrition assistant, manually take over the conversation as a healthcare professional in cases of concerning medical emergency, and capture reusable clinically accurate feedback that shapes future model responses.

## Hosted App

You can access the hosted application to test it directly in your browser [here](https://antenatal-nutrition-chatbot.streamlit.app/).
Alternatively, you can follow instructions below to run locally.

## Prerequisites

- Python 3.12 (recommended – match the local `.venv`)
- An OpenAI API key with access to GPT‑5 models (set `OPENAI_API_KEY`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Set environment variable
```bash
export OPENAI_API_KEY="sk-..." # secret key, please email me if access needed
```

## Run the app

```bash
streamlit run streamlit_app.py
```

Open the URL printed in the terminal (defaults to <http://localhost:8501>).

## Using the app

- **Patient chat (main pane):** Use the chat box at the bottom to send patient questions. The assistant responds with antenatal nutrition guidance, automatically citing knowledge or web sources and obeying guardrails.
- **Agent takeover (sidebar):** The “Healthcare Professionals Only” panel lets you type a manual assistant reply. Click **Send Agent Response** to insert it directly into the chat.
- **Expert clinician feedback:** In the sidebar, the **Provide Feedback** panel shows the most recent assistant response. Add notes (for example, “Always use lots of emojis in your response”) and submit; the guidance is summarized into a reusable rule and appended to the feedback list. Future automated replies incorporate that feedback.

The **Stored Feedback** expander lists all feedback items (newest first) so you can see the rules influencing the model.

## Testing & verification

- **Static check:** `python -m py_compile app.py streamlit_app.py`
- **Manual run-through:** Launch the Streamlit app, send at least two consecutive patient queries, post a clinician takeover response, and submit a feedback note (e.g., “Always use lots of emojis in your response”) to confirm it appears in the sidebar and affects the next automated reply.

