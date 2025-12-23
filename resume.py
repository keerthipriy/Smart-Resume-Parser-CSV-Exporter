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

from dotenv import load_dotenv
import os


load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    st.error(" Please set GEMINI_API_KEY in your .env file")
    st.stop()
os.environ["GOOGLE_API_KEY"] = gemini_key


st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="centered")
st.title("📄 AI-Powered Resume Analyzer & CSV Generator")
st.markdown("Upload a **ZIP file** containing multiple resumes (PDF/DOCX). The AI will extract structured data and generate a downloadable CSV.")

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


prompt_template = """Extract the following information from the resume text below.
Respond ONLY with valid JSON matching this schema (no extra text or markdown):

{format_instructions}

Resume Text:
{resume_text}

If a field is missing, use empty string ("") for strings or empty list ([]) for lists.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["resume_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0.2,           
    max_output_tokens=2048
)

chain = prompt | llm | parser

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        return f"PDF Error: {str(e)}"

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        return f"DOCX Error: {str(e)}"

def extract_text(file_name: str, file_bytes: bytes) -> str:
    if file_name.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif file_name.lower().endswith((".docx", ".doc")):
        return extract_text_from_docx(file_bytes)
    return "Unsupported format"


uploaded_zip = st.file_uploader("Upload ZIP with resumes", type=["zip"])

if uploaded_zip:
    resumes_data = []
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            files = [f for f in z.namelist() if f.lower().endswith((".pdf", ".docx")) and not f.startswith("__MACOSX/")]
            
            progress = st.progress(0)
            status = st.empty()

            for i, file_name in enumerate(files):
                status.text(f"Processing: {file_name} ({i+1}/{len(files)})")
                with z.open(file_name) as f:
                    file_bytes = f.read()

                text = extract_text(file_name, file_bytes)
                if "Error" in text or "Unsupported" in text:
                    resumes_data.append({"File": file_name, "Error": text})
                    continue

                # Truncate very long resumes to avoid token limits
                if len(text) > 25000:
                    text = text[:25000] + "\n\n... (truncated for processing)"

                try:
                    structured: ResumeData = chain.invoke({"resume_text": text})
                    resumes_data.append({
                        "File": file_name,
                        "Name": structured.name,
                        "Email": structured.email,
                        "Phone": structured.phone,
                        "LinkedIn": structured.linkedin,
                        "GitHub": structured.github,
                        "Summary": structured.summary,
                        "Skills": ", ".join(structured.skills),
                        "Experience": " | ".join(structured.experience),
                        "Education": " | ".join(structured.education),
                    })
                except Exception as e:
                    resumes_data.append({"File": file_name, "Error": f"Analysis failed: {str(e)}"})

                progress.progress((i + 1) / len(files))

            status.success("Processing complete!")

    except zipfile.BadZipFile:
        st.error("Invalid or corrupted ZIP file.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    if resumes_data:
        df = pd.DataFrame(resumes_data)
        st.success(f"Processed {len(df)} resumes!")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="resume_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )
