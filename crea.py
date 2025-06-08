import streamlit as st
import requests

API_URL = "http://localhost:8000/api"

st.title("LexiDraft Pro - Document Processor")

st.markdown("""
You can either **upload a text file (.txt)** or **enter/paste your text** below.
If you upload a file, its content will be used; otherwise, the text area input will be used.
""")

# File uploader for .txt files
uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])

text = ""

if uploaded_file is not None:
    try:
        text = uploaded_file.read().decode("utf-8")
        st.text_area("File content", value=text, height=200)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

if not text:
    # If no file or failed to read, allow manual text input
    text = st.text_area("Or enter/paste your text here:", height=200)

if st.button("Process"):
    if not text.strip():
        st.warning("Please upload a file or enter some text before processing.")
    else:
        with st.spinner("Processing..."):
            try:
                payload = {"text": text}

                # Call API endpoints
                endpoints = ["classify", "ner", "summarize", "generate_draft", "detect_risk"]
                responses = {}
                for ep in endpoints:
                    response = requests.post(f"{API_URL}/{ep}", json=payload)
                    response.raise_for_status()
                    responses[ep] = response.json()

                # Display results
                st.subheader("Classification")
                label = responses["classify"].get("label", "N/A")
                confidence = responses["classify"].get("confidence", {})
                st.write(f"**Label:** {label}")
                st.write("**Confidence:**")
                st.json(confidence)

                st.subheader("Named Entity Recognition")
                entities = responses["ner"].get("entities", [])
                if entities:
                    for ent in entities:
                        st.write(f"- {ent}")
                else:
                    st.write("No entities found.")

                st.subheader("Summary")
                summary = responses["summarize"].get("summary", "")
                st.write(summary if summary else "No summary returned.")

                st.subheader("Generated Draft")
                draft = responses["generate_draft"].get("draft_text") or responses["generate_draft"].get("draft") or ""
                st.write(draft if draft else "No draft returned.")

                st.subheader("Risk Detection")
                risk = responses["detect_risk"].get("risk", "N/A")
                confidence_pct = responses["detect_risk"].get("confidence", 0.0)
                severity = responses["detect_risk"].get("severity", "Unknown")
                st.write(f"**Risk:** {risk}")
                st.write(f"**Confidence:** {confidence_pct}%")
                st.write(f"**Severity:** {severity}")

            except requests.exceptions.RequestException as e:
                st.error(f"API request error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
