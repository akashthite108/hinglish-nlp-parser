import json
import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise Omni-Channel NLP & ERP Suite",
    page_icon="🚀",
    layout="centered",
)

st.title("🚀 Enterprise Code-Mixed NLP & Procurement Suite")
st.markdown(
    "Production-grade intelligence engine featuring **Voice Transcription,"
    " Entity Extraction, Risk Scorecards, and ERP Webhooks** for English,"
    " Hindi, and Marathi."
)

# Initialize Session State for Vendor Risk History
if "history" not in st.session_state:
  st.session_state.history = []

# Sidebar Domain Config
st.sidebar.header("🎛️ Enterprise Control Panel")
app_mode = st.sidebar.selectbox(
    "Select Operating Domain:",
    ["B2B Procurement & Vendor Portal", "D2C Customer Support"],
)

# 2. Training Datasets
if app_mode == "B2B Procurement & Vendor Portal":
  train_data = {
      "text": [
          "Bhai 50 bag cement bhej do 350 ke rate par jaldi",
          "Laukar 100 ton steel material site var pathva, project delay hotoy",
          "Rate thoda negotiate karo, dusra vendor kam price de raha hai",
          "PO generate kar do advance payment ho gaya hai",
          "GRN verification pending kyu hai site par",
          "Material quality standard match nahi ho raha rejection batch hai",
          "Supplier invoice mismatch ho gaya GST update karo",
      ],
      "intent": [
          "Procurement RFQ",
          "Urgent Site Dispatch",
          "Vendor Rate Negotiation",
          "PO & Billing Action",
          "Inventory / GRN Audit",
          "Quality Rejection",
          "Tax / GST Compliance",
      ],
  }
else:
  train_data = {
      "text": [
          "Bhai yeh product ka delivery date thoda jaldi ho sakta hai kya",
          "Urgent delivery chahiye jaldi bhejo",
          "Laukar delivery kara, khup mahag aahe he product",
          "Price thoda kam karo na yaar, bhot mehenga hai",
          "Discount milega kya bulk order par",
          "App crash ho raha hai jab payment kar raha hu",
          "Login nahi ho raha error aa raha hai",
          "Ekdum bakwaas service hai yaar ghatiya product",
      ],
      "intent": [
          "Urgent Support",
          "Urgent Support",
          "Urgent Support",
          "Price Negotiation",
          "Price Negotiation",
          "Technical Bug",
          "Technical Bug",
          "Negative Feedback",
      ],
  }

df = pd.DataFrame(train_data)


@st.cache_resource
def train_model(mode_key):
  model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression())
  model.fit(df["text"], df["intent"])
  return model


model = train_model(app_mode)

# 3. Advanced Parsers & Entity Extractors
slang_dict = {
    "bhai": "Colleague / Partner",
    "yaar": "Informal conversational filler",
    "jugaad": "Flexible workaround",
    "bakwaas": "Poor quality",
    "ghatiya": "Low standard",
    "mehenga": "Expensive",
    "jaldi": "Urgent / Fast",
    "laukar": "Quickly (Marathi)",
    "khup": "Very (Marathi)",
    "mahag": "Expensive (Marathi)",
    "site": "Construction / Operational Site",
}


def parse_slang(text):
  found = {}
  for word in text.lower().split():
    cleaned = word.strip(".,!?")
    if cleaned in slang_dict:
      found[cleaned] = slang_dict[cleaned]
  return found


def extract_entities(text):
  """Regex entity extractor for items, quantities, and rates"""
  text_lower = text.lower()
  entities = {"Item": "Not Specified", "Quantity": "N/A", "Target Rate": "N/A"}

  # Extract Item
  known_items = [
      "cement",
      "steel",
      "sand",
      "bricks",
      "iron",
      "tmt",
      "product",
      "app",
      "subscription",
  ]
  for item in known_items:
    if item in text_lower:
      entities["Item"] = item.capitalize()
      break

  # Extract Quantity (e.g., '50 bag', '100 ton', '5 items')
  qty_match = re.search(
      r"(\d+)\s*(bag|bags|ton|tons|kg|pcs|pieces|units)?", text_lower
  )
  if qty_match:
    num, unit = qty_match.groups()
    entities["Quantity"] = f"{num} {unit if unit else 'units'}"

  # Extract Rate/Price (e.g., '350 ke rate', 'rs 500', '350 rs')
  rate_match = re.search(
      r"(?:rate|rs\.?|₹|price)?\s*(\d{2,5})\s*(?:ke rate|rs|rupees)?", text_lower
  )
  if rate_match:
    # Avoid picking up quantity as rate if they are the same
    val = rate_match.group(1)
    if val != qty_match.group(1) if qty_match else True:
      entities["Target Rate"] = f"₹{val}"

  return entities


def calculate_frustration(text):
  score = 20
  if "!" in text or text.isupper():
    score += 30
  if any(
      w in text.lower()
      for w in ["bakwaas", "ghatiya", "delay", "urgent", "reject"]
  ):
    score += 40
  return min(score, 100)


# 4. Feature 1: WhatsApp / Telegram Voice Note Simulator
st.subheader("🎙️ WhatsApp / Telegram Voice Note Simulator")
voice_sim = st.selectbox(
    "Simulate incoming field voice note transcription:",
    [
        "-- Type Manually or Pick Audio Simulation --",
        "🔊 [Audio Clip 1] 'Bhai 50 bag cement bhej do 350 ke rate par jaldi'",
        "🔊 [Audio Clip 2] 'Laukar 100 ton steel material site var pathva, project"
        " delay hotoy'",
        "🔊 [Audio Clip 3] 'App crash ho raha hai jab payment kar raha hu!'",
    ],
)

if "-- Type" not in voice_sim:
  # Extract text inside quotes from the simulation string
  default_input = voice_sim.split("'")[1]
else:
  default_input = ""

user_input = st.text_area(
    "Customer / Vendor Message Stream:",
    value=default_input,
    placeholder="Type or simulate voice input here...",
    height=90,
)

if st.button("Run Full Enterprise Intelligence Suite", type="primary"):
  if user_input.strip():
    prediction = model.predict([user_input])[0]
    confidence = max(model.predict_proba([user_input])[0]) * 100
    slangs = parse_slang(user_input)
    entities = extract_entities(user_input)
    frustration = calculate_frustration(user_input)

    # Log to session history for risk scorecard
    st.session_state.history.append(
        {"Message": user_input, "Intent": prediction, "Frustration": frustration}
    )

    st.divider()

    # Results Layout
    col1, col2 = st.columns(2)
    with col1:
      st.subheader("🎯 Predicted Intent")
      st.info(f"**{prediction}** \n\nConfidence: `{confidence:.1f}%`")

    with col2:
      st.subheader("🌡️ Frustration Index")
      if frustration > 70:
        st.error(f"🔥 **High Risk ({frustration}%)** — Immediate Escalation")
      elif frustration > 40:
        st.warning(f"⚠️ **Moderate Risk ({frustration}%)**")
      else:
        st.success(f"🟢 **Normal Status ({frustration}%)**")

    # Feature 2: Intelligent Entity & Rate Extractor Table
    st.subheader("🏷️ Structured Entity Extraction (Parsed Data)")
    entity_df = pd.DataFrame(list(entities.items()), columns=["Attribute", "Extracted Value"])
    st.table(entity_df)

    # Slang & Context Breakdown
    if slangs:
      st.subheader("💡 Multi-Lingual Regional Slang Decoding")
      for k, v in slangs.items():
        st.markdown(f"- **`{k}`**: {v}")

    # Feature 3: Supplier Risk & Frustration Scorecard Trend
    st.subheader("📈 Vendor / Customer Risk Trend (Session History)")
    if len(st.session_state.history) > 0:
      history_df = pd.DataFrame(st.session_state.history)
      st.line_chart(history_df["Frustration"])

    # Feature 4: Enterprise ERP Webhook & Payload Visualizer
    st.subheader("🔗 Enterprise ERP Webhook JSON Payload")
    erp_payload = {
        "event_id": "EVT-99281-IND",
        "timestamp": "2026-11-03T10:13:00Z",
        "domain": app_mode,
        "classification": {
            "intent": prediction,
            "confidence_score": round(confidence, 2),
            "frustration_index": frustration,
        },
        "extracted_entities": entities,
        "slangs_detected": list(slangs.keys()),
        "automated_action": (
            "Dispatched to SAP/Odoo Procurement Ledger & Priority Queue"
        ),
    }
    with st.expander("📦 Click to view live outgoing JSON payload"):
      st.json(erp_payload)

  else:
    st.warning("Please enter or select a message to analyze.")