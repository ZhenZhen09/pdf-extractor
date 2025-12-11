from flask import Flask, request, render_template, jsonify
import fitz  # PyMuPDF
import os
import json
import io
from PIL import Image
import requests

try:
    import google.generativeai as genai
    GENIE_AVAILABLE = True
except ImportError:
    GENIE_AVAILABLE = False

app = Flask(__name__, template_folder='../templates')

# Read API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GENIE_AVAILABLE and GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Table headers
TABLE_HEADERS = [
    "Customer Name",
    "Transaction Number",
    "Invoice Number",
    "Original/Bal Amount",
    "WHT Amount",
    "Paid Amount (NET)",
    "Description"
]

def pdf_page_to_image(pdf_bytes):
    """Convert the first page of a PDF to a PIL Image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))

def extract_with_gemini(image):
    """Extract table using Google Gemini."""
    if not GENIE_AVAILABLE or not GOOGLE_API_KEY:
        return None
    image.thumbnail((1024, 1024))
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Extract a table with the following columns:
    {', '.join(TABLE_HEADERS)}
    Return valid JSON with key 'table_data'.
    """
    try:
        response = model.generate_content(
            [prompt, image],
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text).get("table_data", [])
    except Exception as e:
        print("Gemini extraction failed:", e)
        return None

def extract_with_groq(image_bytes):
    """Fallback extraction using Groq API."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment.")
    
    url = "https://api.groq.com/openai/v1/responses"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    prompt = f"""
    Extract a table with the following columns:
    {', '.join(TABLE_HEADERS)}
    Return valid JSON with key 'table_data'.
    """
    
    payload = {
        "model": "openai/gpt-oss-20b",  # Groq model
        "input": prompt + "\n" + image_bytes.decode("latin1")
    }
    
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        text = r.json().get("output_text", "")
        return json.loads(text).get("table_data", [])
    except Exception as e:
        print("Groq extraction failed:", e)
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_data():
    files = request.files.getlist('file')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    all_rows = []
    progress_info = []

    for idx, file in enumerate(files):
        try:
            pdf_bytes = file.read()
            image = pdf_page_to_image(pdf_bytes)

            # Try Gemini first
            rows = extract_with_gemini(image)

            # Fallback to Groq
            if rows is None:
                pdf_bytes.seek(0)
                img_buf = io.BytesIO()
                image.save(img_buf, format="PNG")
                img_buf.seek(0)
                rows = extract_with_groq(img_buf.read())

            all_rows.extend(rows)
            percent = int(((idx + 1) / len(files)) * 100)
            progress_info.append({"file": file.filename, "progress": percent})

        except Exception as e:
            print(f"Failed to process {file.filename}: {e}")
            progress_info.append({"file": file.filename, "progress": -1})

    return jsonify({"table_data": all_rows, "file_progress": progress_info})

if __name__ == "__main__":
    app.run(debug=True)
