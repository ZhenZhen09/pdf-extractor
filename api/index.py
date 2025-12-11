from flask import Flask, request, render_template, jsonify
import fitz  # PyMuPDF
import os
import json
import io
import base64
import requests
from PIL import Image

app = Flask(__name__, template_folder='../templates')

# Groq API Key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
    return img_data  # Keep as bytes for Base64 conversion

def extract_table_with_groq(image_bytes):
    """Extract table from image using Groq API."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")

    url = "https://api.groq.com/openai/v1/responses"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Convert image bytes to base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = f"""
Extract a table with the following columns:
{', '.join(TABLE_HEADERS)}
The image is provided in Base64 format. Return valid JSON
with a key "table_data" containing a list of row objects.
"""

    payload = {
        "model": "openai/gpt-oss-20b",
        "input": prompt + f"\n[IMAGE_BASE64]{image_b64}[/IMAGE_BASE64]"
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
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        pdf_bytes = file.read()
        image_bytes = pdf_page_to_image(pdf_bytes)
        table_rows = extract_table_with_groq(image_bytes)

        return jsonify({
            "table_data": table_rows,
            "file_progress": [{"file": file.filename, "progress": 100}]
        })

    except Exception as e:
        print("Error processing PDF:", e)
        return jsonify({"error": "Processing failed: " + str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
