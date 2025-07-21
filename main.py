import os
import json
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pypdf import PdfReader
from resumeparser import ats_extractor

UPLOAD_DIR = "__DATA__"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensures the folder exists

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


def extract_text(path: str) -> str:
    """Extract text from a PDF file."""
    content = []
    reader = PdfReader(path)
    for page in reader.pages:
        content.append(page.extract_text() or "")
    return "\n".join(content)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process")
async def process_resumes(
    request: Request,
    resume_files: list[UploadFile] = File(default=[]),
    resume_text: str = Form(default=""),
    output_dir: str = Form(default=""),
):
    """Process resumes either from uploaded files or pasted text."""
    results = []
    is_single = bool(resume_text.strip()) or len(resume_files) == 1

    # ========== Default Output Path (Same as Input Path) ==========
    resolved_output_dir = output_dir.strip() or None  # If output_dir is empty, resolved_output_dir is None

    # If output_dir is provided, make sure it exists
    if resolved_output_dir:
        os.makedirs(resolved_output_dir, exist_ok=True)
    else:
        resolved_output_dir = None  # Default to None (same as input file location)

    # ========== Case 1: Single Resume Text ==========
    if resume_text.strip():
        try:
            # Process the pasted resume text
            json_result = ats_extractor(resume_text)
            parsed = json.loads(json_result)

            # Save parsed result to output directory (default or user-specified)
            output_file = os.path.join(resolved_output_dir or UPLOAD_DIR, f"processed_pasted_resume.json")
            with open(output_file, "w") as f:
                json.dump(parsed, f, indent=2)

            return templates.TemplateResponse("index.html", {
                "request": request,
                "single_result": parsed,
                "batch_result": None,
                "output_path": output_file
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse pasted resume: {e}")

    # ========== Case 2: Multiple Uploaded Files ==========
    success, failure = [], []

    for file in resume_files:
        filename = file.filename
        local_path = os.path.join(UPLOAD_DIR, filename)

        # Save the uploaded file to the server
        with open(local_path, "wb") as f:
            f.write(await file.read())

        try:
            # Extract resume text and process it
            resume_text = extract_text(local_path)
            json_result = ats_extractor(resume_text)
            parsed_json = json.loads(json_result)

            # Resolve output file path in the specified directory
            input_parent_dir = str(Path(local_path).parent)
            output_path = os.path.join(resolved_output_dir or input_parent_dir, f"processed_{Path(filename).stem}.json")

            # Save parsed result
            with open(output_path, "w") as f:
                json.dump(parsed_json, f, indent=2)

            success.append({
                "filename": filename,
                "status": "✅ Processed",
                "download": f"/download/{Path(output_path).name}",
                "output_path": output_path
            })
        except Exception as e:
            failure.append({
                "filename": filename,
                "status": "❌ Failed",
                "error": str(e)
            })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "single_result": None,
        "batch_result": {
            "success": success,
            "failure": failure,
            "total": len(resume_files),
            "output_dir": resolved_output_dir or UPLOAD_DIR
        }
    })


@app.get("/download/{filename}")
def download_json(filename: str):
    """Download the processed JSON file."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type="application/json", filename=filename)
