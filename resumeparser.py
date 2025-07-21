import os
import json
import yaml
from pathlib import Path
from typing import Union
from openai import AzureOpenAI
from pypdf import PdfReader
from docx import Document 
from datetime import datetime

# Load Azure OpenAI configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

api_key = config["AZURE_OPENAI_API_KEY"]
api_endpoint = config["AZURE_OPENAI_ENDPOINT"]
api_version = config["AZURE_OPENAI_API_VERSION"]
deployment_name = config["AZURE_OPENAI_DEPLOYMENT_NAME"]

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=api_endpoint,
    api_version=api_version
)

# ------------------ Core AI Parsing (untouched) ------------------ #
from datetime import datetime

def ats_extractor(resume_text: str) -> str:
    CURRENT_MONTH = datetime.now().strftime("%B %Y")  # e.g., June 2025

    example_output = """
{
  "full_name": "John Doe",
  "email_id": "john.doe@example.com",
  "mobile": "+1-123-456-7890",
  "location": "New York, NY",
  "resume_summary": "John Doe is an experienced DevOps Engineer with 7 years of expertise in building reliable CI/CD pipelines and scalable cloud infrastructure. He has worked with tools such as Terraform, Docker, and Kubernetes across multiple enterprise environments. John’s recent accomplishments include leading a monitoring system rollout and optimizing deployment pipelines. Known for his strong collaboration and systems thinking, he brings both technical depth and team alignment to his work.",
  "number_of_years_of_experience": 8,
  "current_designation": "Lead DevOps Engineer",
  "current_company": "CloudX",
  "employment_details": [
    {
      "company": "CloudX",
      "role": "DevOps Engineer",
      "duration": "Feb 2021 - Present"
    },
    {
      "company": "Atlanta Regional Commission",
      "role": "Full Stack Web Developer",
      "duration": "July 2020 - Jan 2021"
    }
  ],
  "skills": [
    "Python", "Docker", "Kubernetes", "Terraform", "AWS",
    "CI/CD", "Monitoring", "Problem-solving", "Team collaboration"
  ],
  "projects": [
    {
      "title": "CI/CD Automation",
      "summary": "Built an automated pipeline using GitHub Actions and Terraform for cloud deployments."
    }
  ],
  "certifications": [
    "AWS Certified Solutions Architect – Associate"
  ],
  "education": [
    {
      "dates": "2015 - 2019",
      "degree": "B.S. in Computer Science",
      "institution": "University of Colorado"
    }
  ],
  "publications": [
    "Optimizing Kubernetes for High Availability (IEEE, 2023)"
  ],
  "patents": [
    "US12345678B2 - Cloud-native autoscaling engine"
  ],
  "social_profiles": {
    "linkedin": "https://linkedin.com/in/johndoe",
    "facebook": "https://facebook.com/johndoe.dev",
    "github": "https://github.com/johndoe"
  }
}
"""

    system_prompt = f"""
You are an expert AI resume Analyser and Parser.
Today's date is {CURRENT_MONTH}.

From the plain text resume below, extract the following fields and return ONLY a valid JSON object:

1. full_name  
2. email_id  
3. mobile — No spaces, No parentheses or periods.  
4. location — (city or region)  
5. resume_summary — A professional 4–5 sentence summary written like a recruiter  
6. number_of_years_of_experience — Sum up the durations of all roles listed in employment_details.  
   - For "Present", "Current", or "Till Date", use {CURRENT_MONTH} internally as the end date.  
   - **Sum all job durations independently, even if they overlap in time.**  
   - Round the total to the nearest whole year.  
7. current_designation — Most recent job title based on the latest start or end date.  
   - If the job is marked as "Present", "Current", or "Till Date", it is also the user's current role.  
8. current_company — Extract the company name from the role marked as "Present", "Current", or "Till Date"  
9. employment_details — list of:
    - company
    - role
    - duration (e.g., "Jan 2017 - Present") — Use `-` only, never en dashes or em dashes  
10. skills — technical + soft skills as one flat list  
11. projects — Major 1–3 projects with:
    - title
    - summary (1 line)  
12. certifications — List all certifications mentioned (e.g., "Google Ads Certified", "AWS Certified Cloud Practitioner")  
13. education — list of:
    - dates (e.g., "2018 - 2022")
    - degree
    - institution  
14. publications — List of any research publications or articles authored  
15. patents — List any patents mentioned  
16. social_profiles — An object with:
    - linkedin
    - facebook
    - github

⚠️ STRICT RULES:
- DO NOT replace "Present", "Till Date", or "Current" in job durations with any specific month or year.
- Use today's date only for **internal calculation** of experience (not for display).
- DO NOT merge overlapping job durations — sum each job’s total time separately even if they overlap.
- DO NOT invent or hallucinate any jobs, companies, roles, or projects not explicitly stated.
- If certain fields are missing, return them as empty strings or empty lists.
- Use a standard hyphen - for date ranges instead of an en dash or special character.
- Donot have spaces or gaps in phone or mobile numbers, only expected separators like "+", "-" 
- Return **only valid JSON** — no markdown, no extra comments.

Example output format:

{example_output}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": resume_text},
    ]

    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        temperature=0.1,
        max_tokens=3000,
    )

    return response.choices[0].message.content


# ------------------ File Parsing & Batch Processing ------------------ #

def extract_text(file_path: Path) -> str:
    """Extract text from PDF, DOCX, or TXT files safely."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        try:
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {file_path.name}. Error: {str(e)}")

    elif suffix == ".docx":
        try:
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX: {file_path.name}. Error: {str(e)}")

    elif suffix == ".txt":
        return file_path.read_text()

    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def process_resume_file(file_path: Union[str, Path], output_dir: Union[str, Path] = None):
    """Extract, parse, and write output JSON next to file"""
    file_path = Path(file_path)
    output_dir = Path(output_dir) if output_dir else file_path.parent

    resume_text = extract_text(file_path)
    parsed_json = ats_extractor(resume_text)

    output_path = output_dir / f"processed_{file_path.stem}.json"
    #output_path.write_text(parsed_json, encoding="utf-8")
    output_path.write_text(json.dumps(parsed_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Saved: {output_path}")


def batch_process_dir(input_dir: Union[str, Path], output_dir: Union[str, Path] = None):
    """Process all resumes in a directory"""
    input_dir = Path(input_dir)
    for file in input_dir.glob("*"):
        if file.suffix.lower() in [".pdf", ".docx", ".txt"]:
            try:
                process_resume_file(file, output_dir=output_dir)
            except Exception as e:
                print(f"❌ Failed to process {file.name}: {e}")
