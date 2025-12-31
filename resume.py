import streamlit as st
import zipfile
import io
import pandas as pd
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

import PyPDF2
from docx import Document
import os

# ========================
# Secure API Key Handling (for Streamlit Cloud)
# ========================
gemini_key = st.secrets.get("GEMINI_API_KEY")

if not gemini_key:
    st.error("GEMINI_API_KEY is not set. Please add it in App Secrets on Streamlit Cloud.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = gemini_key

# ========================
# Page Config & UI
# ========================
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="centered")
st.title("📄 AI-Powered Resume Analyzer & CSV Generator")
st.markdown("""
Upload a **ZIP file** containing multiple resumes (PDF or DOCX).  
The AI (Gemini) will extract structured data from each resume and generate a downloadable **CSV**.
""")

# ========================
# Pydantic Schema for Structured Output
# ========================
class ResumeData(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    linkedin: str = Field(default="", description="LinkedIn URL (empty if none)")
    github: str = Field(default="", description="GitHub URL (empty if none)")
    summary: str = Field(description="Professional summary or objective")
    skills: List[str] = Field(description="List of technical and soft skills")
    experience: List[str] = Field(description="List of work experience entries (company, role, duration)")
    education: List[str] = Field(description="List of education entries (degree, institution, year)")

parser = PydanticOutputParser(pydantic_object=ResumeData)

# ========================
# Prompt Template
# ========================
prompt_template = """Extract the following information from the resume text below.
Respond ONLY with valid JSON matching the schema (no extra text, explanations, or markdown):

{format_instructions}

Resume Text:
{resume_text}

Rules:
- If information is missing, use "" for strings and [] for lists.
- Do not add explanations or formatting.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["resume_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# ========================
# LLM Setup (Correct & Stable Model)
# ========================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",        # Stable, fast, and reliable as of Dec 2025
    temperature=0.2,
    max_output_tokens=2048,
)

chain = prompt | llm | parser

# ========================
# Text Extraction Functions
# ========================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception as e:
        return f"[PDF Extraction Error: {str(e)}]"

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        return f"[DOCX Extraction Error: {str(e)}]"

def extract_text(file_name: str, file_bytes: bytes) -> str:
    if file_name.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif file_name.lower().endswith((".docx", ".doc")):
        return extract_text_from_docx(file_bytes)
    else:
        return "[Unsupported file format]"

# ========================
# Main App Logic
# ========================
uploaded_zip = st.file_uploader("Upload ZIP file with resumes (PDF/DOCX)", type=["zip"])

if uploaded_zip:
    resumes_data = []

    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            files = [
                f for f in z.namelist()
                if f.lower().endswith((".pdf", ".docx")) and not f.startswith("__MACOSX/")
            ]

            if not files:
                st.warning("No valid PDF or DOCX files found in the ZIP.")
                st.stop()

            st.info(f"Found {len(files)} resume(s). Starting processing...")

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file_name in enumerate(files):
                status_text.text(f"Processing {i+1}/{len(files)}: {file_name}")

                with z.open(file_name) as f:
                    file_bytes = f.read()

                raw_text = extract_text(file_name, file_bytes)

                if raw_text.startswith("["):
                    resumes_data.append({
                        "File": file_name,
                        "Name": "", "Email": "", "Phone": "", "LinkedIn": "", "GitHub": "",
                        "Summary": "", "Skills": "", "Experience": "", "Education": "",
                        "Error": raw_text
                    })
                    progress_bar.progress((i + 1) / len(files))
                    continue

                # Truncate long resumes safely
                max_chars = 20000
                if len(raw_text) > max_chars:
                    half = max_chars // 2
                    text = raw_text[:half] + "\n\n[... truncated ...]\n\n" + raw_text[-half:]
                else:
                    text = raw_text

                try:
                    structured: ResumeData = chain.invoke({"resume_text": text})

                    resumes_data.append({
                        "File": file_name,
                        "Name": structured.name or "",
                        "Email": structured.email or "",
                        "Phone": structured.phone or "",
                        "LinkedIn": structured.linkedin or "",
                        "GitHub": structured.github or "",
                        "Summary": structured.summary or "",
                        "Skills": ", ".join(structured.skills) if structured.skills else "",
                        "Experience": " | ".join(structured.experience) if structured.experience else "",
                        "Education": " | ".join(structured.education) if structured.education else "",
                        "Error": ""
                    })

                except Exception as e:
                    resumes_data.append({
                        "File": file_name,
                        "Name": "", "Email": "", "Phone": "", "LinkedIn": "", "GitHub": "",
                        "Summary": "", "Skills": "", "Experience": "", "Education": "",
                        "Error": f"AI Parsing Failed: {str(e)[:100]}"
                    })

                progress_bar.progress((i + 1) / len(files))

            status_text.success("All resumes processed!")

    except zipfile.BadZipFile:
        st.error("Invalid or corrupted ZIP file.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.stop()

    # ========================
    # Display Results & CSV Download
    # ========================
    if resumes_data:
        df = pd.DataFrame(resumes_data)
        columns_order = ["File", "Name", "Email", "Phone", "LinkedIn", "GitHub",
                         "Summary", "Skills", "Experience", "Education", "Error"]
        df = df[columns_order]

        st.success(f"Successfully processed {len(df)} resume(s)!")
        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="resume_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )

        failed = df["Error"].astype(bool).sum()
        if failed > 0:
            st.warning(f"{failed} resume(s) had errors. Check the 'Error' column.")
