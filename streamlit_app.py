import os
import asyncio
import streamlit as st
from openai import AsyncOpenAI

from app import run_workflow, WorkflowInput

API_KEY = st.secrets["OPENAI_API_KEY"]
if API_KEY:
  os.environ["OPENAI_API_KEY"] = API_KEY
else:
  st.warning(
    "No OpenAI API key found. Please add `OPENAI_API_KEY` to `.streamlit/secrets.toml`."
  )


def rerun_app():
  if hasattr(st, "rerun"):
    st.rerun()
  elif hasattr(st, "experimental_rerun"):
    st.experimental_rerun()


st.set_page_config(page_title="Antenatal Nutrition Chatbot 🤱🏻", layout="wide")

st.markdown(
  """
  <style>
    [data-testid="stSidebar"] {
      width: 800px;
    }
  </style>
  """,
  unsafe_allow_html=True,
)

GENERATE_FEEDBACK_PROMPT = """You are an antenatal care specialist who creates concise guidance for a digital nutrition assistant.

Instructions:
- You will receive the assistant's most recent response to a patient plus clinician notes critiquing that response (e.g., incorrect advice, missing escalation, tone issues).
- Turn those notes into a short, general rule (maximum one sentence) that the assistant can apply to similar future messages.
- Make the guidance general, actionable and easy for a junior clinician or support worker to follow.
- If the feedback is generic and not directly related to the content of the assistant's recent response e.g. "Respond in 2 sentences or less", then respond in 2 sentences, do not consider the content of the assistant's recent response when generating the feedback.

Always return feedback in 1 sentence or less and follow the clinician notes exactly.

Example 1:
Last assistant message: "It sounds like a normal headache. Try to rest and drink more water."
Clinician notes: "Dismisses red-flag symptoms; patient mentioned sudden severe headache with blurred vision—needs urgent escalation."
Feedback: "Treat sudden severe headaches with vision changes as an emergency and direct the patient to immediate clinical care for possible preeclampsia."

Example 2:
Last assistant message: "You should cut your portions so you don't gain too much weight."
Clinician notes: "Advice is shaming and ignores balanced nutrition; encourage supportive tone and practical planning."
Feedback: "Use supportive language and focus on balanced meals, portion guidance, and empathetic reassurance rather than weight shaming."

Example 2:
Last assistant message: "I’m sorry you’re struggling with body image after birth — that can feel really hard and you’re not alone; I'm sorry, but I cannot answer questions outside the scope of antenatal nutritional care. For support with body image and postpartum mental health, you may find helpful information at Postpartum Support International, the NHS Pregnancy and Baby pages, the Mayo Clinic’s postpartum mental health section, and MotherToBaby; if you need crisis help, contact local emergency services or a mental health crisis line. Would you like help with nutrition-related concerns after birth (meal ideas, healthy weight-loss guidance, or breastfeeding nutrition)?"
Clinician notes: "Respond in 2 sentences or less"
Feedback: "Keep responses concise; always respond in 2 sentences or less"
"""

GPT_FEEDBACK_MODEL = "gpt-5-mini"
DEFAULT_FEEDBACK = "When asked about medical questions or emergency ONLY, prompt the user to wait until we connect them to a healthcare professional. Do NOT prompt this for general nutrition questions."

feedback_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

async def _generate_feedback_async(assistant_message: str, clinician_notes: str) -> str:
  response = await feedback_client.responses.create(
    model=GPT_FEEDBACK_MODEL,
    input=[
      {
        "role": "system",
        "content": GENERATE_FEEDBACK_PROMPT
      },
      {
        "role": "user",
        "content": f"Last assistant message:\n{assistant_message}\n\nClinician notes:\n{clinician_notes}"
      }
    ],
    reasoning={
      "effort": "minimal"
    },
    text={
        "verbosity": "low"
    }
  )
  return response.output_text.strip()


def generate_feedback(assistant_message: str, clinician_notes: str) -> str:
  return asyncio.run(_generate_feedback_async(assistant_message, clinician_notes))


st.title("Antenatal Nutrition Chatbot")

# Configure API key for underlying OpenAI client used by agents
if API_KEY:
  os.environ["OPENAI_API_KEY"] = API_KEY

if "messages" not in st.session_state:
  st.session_state.messages = []

if "agent_input" not in st.session_state:
  st.session_state.agent_input = ""

if "reset_agent_input" not in st.session_state:
  st.session_state.reset_agent_input = False

if "expert_clinician_feedback" not in st.session_state:
  st.session_state.expert_clinician_feedback = [DEFAULT_FEEDBACK]

if "agent_feedback_notes" not in st.session_state:
  st.session_state.agent_feedback_notes = ""

if "reset_feedback_inputs" not in st.session_state:
  st.session_state.reset_feedback_inputs = False

if st.session_state.reset_agent_input:
  st.session_state.agent_input = ""
  st.session_state.reset_agent_input = False

if st.session_state.reset_feedback_inputs:
  st.session_state.agent_feedback_notes = ""
  st.session_state.reset_feedback_inputs = False

feedback_lines = [f"- {item}" for item in st.session_state.expert_clinician_feedback]
feedback_str = "\n".join(feedback_lines)

prompt = st.chat_input("Type your question here...")

if prompt:
  st.session_state.messages.append({"role": "user", "content": prompt})

  try:
    result = asyncio.run(
      run_workflow(
        WorkflowInput(
          input_as_text=prompt,
          conversation_history=st.session_state.messages,
          clinician_feedback=feedback_str
        )
      )
    )
    if isinstance(result, dict):
      response_text = result.get("message") or result.get("output_text") or str(result)
      print(result)
      print(response_text)
    else:
      response_text = str(result)
      print(result)
      print(response_text)
  except Exception as e:
    response_text = f"Error: {e}"

  st.session_state.messages.append({"role": "assistant", "content": response_text})
  rerun_app()

st.sidebar.header("Agent Console — Healthcare Professionals Only")
st.sidebar.markdown(
  "Use this console to step in and respond directly to the patient. Messages sent here "
  "appear in the conversation as assistant responses. You can also submit general feedback to improve the agent's responses."
)
st.sidebar.text_area(
  "Draft agent response",
  key="agent_input",
  placeholder="Enter a response for the patient...",
)
if st.sidebar.button("Send Agent Response", use_container_width=True, key="agent_send"):
  response = st.session_state.agent_input.strip()
  if response:
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.reset_agent_input = True
    rerun_app()

with st.sidebar.expander("Provide Feedback", expanded=False):
  st.markdown(
    "Review the latest assistant response and record guidance to shape future replies."
  )
  last_assistant_message = None
  for message in reversed(st.session_state.messages):
    if message.get("role") == "assistant":
      last_assistant_message = message.get("content", "")
      break
  st.text_area(
    "Clinician notes",
    key="agent_feedback_notes",
    placeholder="Describe how the assistant should adjust future responses...",
  )
  if st.button("Submit Feedback", use_container_width=True, key="submit_feedback"):
    clinician_notes = st.session_state.agent_feedback_notes.strip()
    if not last_assistant_message:
      st.warning("Feedback requires at least one assistant response to review.")
    elif not clinician_notes:
      st.warning("Please provide clinician notes before submitting feedback.")
    else:
      try:
        feedback_text = generate_feedback(last_assistant_message, clinician_notes)
      except Exception as error:
        st.warning(f"Feedback generation failed: {error}")
      else:
        st.session_state.expert_clinician_feedback.append(feedback_text)
        st.session_state.reset_feedback_inputs = True
        rerun_app()

with st.sidebar.expander("Stored Feedback", expanded=True):
  for item in st.session_state.expert_clinician_feedback:
    st.markdown(f"- {item}")

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])
