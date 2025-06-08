import streamlit as st
from pymongo import MongoClient
import bcrypt
import requests
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re

# ----- MongoDB Connection -----
MONGO_URI = "mongodb+srv://yashmalviya2304:yash0539s@cluster0.enyverb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["lex"]
users_collection = db["users"]

API_URL = "http://localhost:8000/api"

# ----- Auth Session State -----
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ----- Password Hashing -----
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

# ----- Registration/Login Logic -----
def register_user(username, password):
    if users_collection.find_one({"username": username}):
        return False, "Username already exists!"
    users_collection.insert_one({
        "username": username,
        "password": hash_password(password)
    })
    return True, "Registration successful!"

def login_user(username, password):
    user = users_collection.find_one({"username": username})
    if not user:
        return False, "User not found!"
    if check_password(password, user["password"]):
        return True, "Login successful!"
    return False, "Incorrect password."

# ----- Theme Toggle -----
theme = st.sidebar.radio("🌗 Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
            body, .block-container {
                background-color: #1e1e1e;
                color: #f0f0f0;
            }
            .stTextInput>div>div>input {
                background-color: #333;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

# ----- Login UI -----
if not st.session_state.authenticated:
    st.title("🔐 LexiDraft - Login / Sign Up")
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            success, msg = login_user(username, password)
            st.info(msg)
            if success:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.experimental_rerun()

    with tab2:
        new_user = st.text_input("New Username", key="signup_user")
        new_pass = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            success, msg = register_user(new_user, new_pass)
            st.info(msg)

    st.stop()

# ----- Logout -----
st.sidebar.success(f"👤 Logged in as: {st.session_state.username}")
if st.sidebar.button("🔓 Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.experimental_rerun()

# ---------- HEADER ----------
st.markdown("""
<h1 style="font-size: 3rem; font-weight: 800; color: #2c3e50;">
LexiDraft Pro - Legal Document Analyzer 📑
</h1>
""", unsafe_allow_html=True)

# ---------- SIDEBAR INPUT ----------
with st.sidebar:
    st.header("Upload or Paste Text")
    uploaded_file = st.file_uploader("📄 Upload a .txt file", type=["txt"])
    text_input = ""
    if uploaded_file:
        try:
            text_input = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    if not text_input:
        text_input = st.text_area("✍️ Or paste your text:", height=200)

    # Token stats
    char_count = len(text_input)
    word_count = len(re.findall(r"\w+", text_input))
    st.markdown(f"🧮 **Characters**: `{char_count}` | **Tokens**: `{word_count}`")

    process_btn = st.button("🚀 Process Document")

# ---------- UTILITIES ----------
def render_badge(text, severity="info"):
    colors = {
        "low": "#28a745", "medium": "#ffc107",
        "high": "#dc3545", "info": "#007bff"
    }
    color = colors.get(severity, "#6c757d")
    return f'<span style="background:{color};padding:4px 12px;border-radius:12px;color:white;font-weight:600;">{text}</span>'

def export_to_pdf(text, filename="output.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    lines = text.split("\n")
    y = height - 40
    for line in lines:
        c.drawString(40, y, line[:100])
        y -= 15
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    buffer.seek(0)
    return buffer

# ---------- PROCESSING ----------
if process_btn:
    if not st.session_state.authenticated:
        st.warning("🔐 Please log in to process the document.")
        st.stop()
    
    if not text_input.strip():
        st.warning("⚠️ Please upload a file or paste some text.")
        st.stop()

    try:
        with st.spinner("Analyzing document..."):
            payload = {"text": text_input}
            endpoints = ["classify", "ner", "summarize", "generate_draft", "detect_risk"]
            responses = {ep: requests.post(f"{API_URL}/{ep}", json=payload).json() for ep in endpoints}

        tabs = st.tabs(["📄 Classification", "🧠 NER", "✍️ Summary", "📑 Draft", "⚠️ Risk Detection"])

        with tabs[0]:
            label = responses["classify"].get("label", "N/A")
            conf = responses["classify"].get("confidence", {}).get("score", 0.0)
            st.markdown(f"**Label**: {label}")
            st.markdown(f"**Confidence**: {render_badge(f'{int(conf*100)}%', 'info')}", unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("**Named Entities:**")
            entities = responses["ner"].get("entities", [])
            if entities:
                st.markdown("<ul>" + "".join([f"<li>{e}</li>" for e in entities]) + "</ul>", unsafe_allow_html=True)
            else:
                st.info("No entities found.")

        with tabs[2]:
            summary = responses["summarize"].get("summary", "")
            if summary:
                st.text_area("📋 Summary Output", value=summary, height=300)
                pdf = export_to_pdf(summary, "summary.pdf")
                st.download_button("⬇️ Download Summary PDF", pdf, file_name="summary.pdf", mime="application/pdf")
            else:
                st.info("No summary returned.")

        with tabs[3]:
            draft = responses["generate_draft"].get("draft_text", "")
            if draft:
                st.text_area("📝 Generated Draft", value=draft, height=400)
                pdf = export_to_pdf(draft, "draft.pdf")
                st.download_button("⬇️ Download Draft PDF", pdf, file_name="draft.pdf", mime="application/pdf")
            else:
                st.info("No draft returned.")

        with tabs[4]:
            risk = responses["detect_risk"].get("risk", "N/A")
            confidence = responses["detect_risk"].get("confidence", 0.0)
            severity = responses["detect_risk"].get("severity", "Unknown").lower()
            st.markdown(f"**Risk:** {risk}")
            st.markdown(f"**Confidence:** {render_badge(f'{int(confidence*100)}%', severity)}", unsafe_allow_html=True)
            st.markdown(f"**Severity:** {render_badge(severity.title(), severity)}", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
