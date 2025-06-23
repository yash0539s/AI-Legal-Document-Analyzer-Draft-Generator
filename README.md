# LexiDraft Pro – AI-Powered Legal Document Analyzer & Draft Generator

LexiDraft Pro is an AI-enabled legal document analysis platform designed to streamline and automate the understanding, classification, summarization, and drafting of legal documents. It provides a secure, interactive web interface for legal professionals to upload documents, gain insights, and generate drafts in real time.

## Features

- **Document Upload & OCR**: Extract text from scanned legal documents using Tesseract OCR.
- **Legal Document Classification**: Classify documents into predefined legal types using LegalBERT.
- **Named Entity Recognition (NER)**: Identify legal-specific entities within the text.
- **Summarization**: Extract concise summaries from long legal texts using transformer-based models.
- **Draft Generation**: Automatically generate legal drafts from extracted content using GPT-2.
- **Risk Detection**: Evaluate legal risks with scoring and severity tagging.
- **User Authentication**: Secure login/signup with MongoDB integration.
- **PDF Export**: Download generated drafts and summaries as PDF.
- **Download & Share**: Share or store results easily.
- **Optional Versioning**: Add version history for individual users.

## Tech Stack

**Frontend**
- Streamlit (Responsive UI with dynamic tabs)
- HTML/CSS (Custom styling)
- Download and Export integrations (PDF via ReportLab)

**Backend (API)**
- FastAPI (Modular architecture)
- Tesseract OCR (Image-to-text conversion)
- LegalBERT, GPT-2 (Document classification and draft generation)
- ML models exposed via RESTful APIs

**MLOps & Deployment**
- MLflow (Experiment tracking)
- Hydra (Dynamic config management)
- Docker (Containerized deployment)
- GitHub Actions (CI/CD pipeline)
- MongoDB (User auth and data persistence)

## How to Run

### Prerequisites

- Python 3.8+
- MongoDB Atlas or Local MongoDB instance
- Docker (optional for containerized deployment)

### Setup

```bash
git clone https://github.com/your-username/LexiDraft-Pro.git
cd LexiDraft-Pro

# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (in another terminal)
cd ../streamlit_ui
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Folder Structure

```
LexiDraft-Pro/
│
├── backend/
│   ├── api/
│   ├── core/
│   ├── models/
│   └── main.py
│
├── streamlit_ui/
│   └── streamlit_app.py
│
├── README.md
└── requirements.txt
```

## License

This project is licensed under the MIT License.