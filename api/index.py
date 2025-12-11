from flask import Flask, request, render_template, jsonify
import fitz  # PyMuPDF
import os
import io
from PIL import Image
import requests
import json

app = Flask(__name__, template_folder='../templates')

# Read Groq API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment.")

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

def pdf_pages_to_images(pdf_bytes):
    """Convert all pages of a PDF to a list of PIL Images."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        images.append(Image.open(io.BytesIO(img_data)))
    return images

def extract_table_with_groq(image_bytes):
    """Call Groq API to extract structured table from an image."""
    url = "https://api.groq.com/openai/v1/responses"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""
Extract a table with the following columns:
{', '.join(TABLE_HEADERS)}
Return valid JSON with a key "table_data" containing a list of row objects.
"""
    payload = {
        "model": "openai/gpt-oss-20b",
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
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    all_rows = []

    try:
        pdf_bytes = file.read()
        images = pdf_pages_to_images(pdf_bytes)

        for image in images:
            # Resize for faster processing
            image.thumbnail((1024, 1024))
            img_buf = io.BytesIO()
            image.save(img_buf, format="PNG")
            img_buf.seek(0)

            rows = extract_table_with_groq(img_buf.read())
            all_rows.extend(rows)

        return jsonify({"table_data": all_rows})

    except Exception as e:
        print(f"Failed to process {file.filename}: {e}")
        return jsonify({"error": "Processing failed: " + str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
