from flask import Flask, request, render_template, jsonify
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import json
import io
from PIL import Image
import requests  # For Groq fallback

app = Flask(__name__, template_folder='../templates')

# Configure API keys
GENIE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = "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"

genai.configure(api_key=GENIE_API_KEY)

# Target table headers
TABLE_HEADERS = [
    "Customer Name",
    "Transaction Number",
    "Invoice Number",
    "Original/Bal Amount",
    "WHT Amount",
    "Paid Amount (NET)",
    "Description"
]

def pdf_page_to_image(pdf_stream):
    """Convert first page of PDF to PIL Image."""
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))

def extract_table_from_image_gemini(image):
    """Try extracting table from image using Google Gemini."""
    image.thumbnail((1024, 1024))
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Extract table data with the following columns:
    {', '.join(TABLE_HEADERS)}
    Return valid JSON with key "table_data" containing a list of row objects.
    """
    try:
        response = model.generate_content(
            [prompt, image],
            generation_config={"response_mime_type": "application/json"}
        )
        result = json.loads(response.text)
        return result.get("table_data", [])
    except Exception as e:
        print("Gemini extraction failed:", e)
        return None

def extract_table_from_image_groq(image_bytes):
    """Fallback extraction using Groq API."""
    url = "https://api.groq.com/v1/extract"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input_type": "image",
        "image_base64": image_bytes.decode("latin1"),
        "columns": TABLE_HEADERS
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("table_data", [])
    except Exception as e:
        print("Groq fallback failed:", e)
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
    file_progress = []

    for idx, file in enumerate(files):
        try:
            pdf_bytes = file.read()
            image = pdf_page_to_image(pdf_bytes)

            # Try Google Gemini first
            rows = extract_table_from_image_gemini(image)

            # Fallback to Groq if Gemini fails
            if rows is None:
                pdf_bytes.seek(0)
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                rows = extract_table_from_image_groq(img_bytes.read())

            all_rows.extend(rows)
            percent = int(((idx + 1) / len(files)) * 100)
            file_progress.append({"file": file.filename, "progress": percent})

        except Exception as e:
            print(f"Failed to process {file.filename}: {e}")
            file_progress.append({"file": file.filename, "progress": -1})

    return jsonify({"table_data": all_rows, "file_progress": file_progress})

if __name__ == "__main__":
    app.run(debug=True)
