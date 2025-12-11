from flask import Flask, request, render_template, jsonify
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import json
import io
from PIL import Image

app = Flask(__name__, template_folder='../templates')

# Configure Google Gemini API Key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

def pdf_page_to_image(pdf_stream):
    """Convert first page of PDF to PIL Image."""
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))

def extract_table_from_image(image):
    """Send image to Gemini API and extract table data."""
    image.thumbnail((1024, 1024))  # Resize for faster processing

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = """
    Analyze this image and extract the following table columns:
    Customer Name | Transaction Number | Invoice Number | Original/Bal Amount | WHT Amount | Paid Amount (NET) | Description
    Output must be a valid JSON with a key "table_data" containing a list of row objects.
    """

    response = model.generate_content(
        [prompt, image],
        generation_config={"response_mime_type": "application/json"}
    )

    result = json.loads(response.text)
    return result.get("table_data", [])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    files = request.files.getlist('file')
    if not files:
        return jsonify({"error": "No files selected"}), 400

    all_rows = []

    try:
        for file in files:
            pdf_bytes = file.read()
            image = pdf_page_to_image(pdf_bytes)
            table_rows = extract_table_from_image(image)
            all_rows.extend(table_rows)

        return jsonify({"table_data": all_rows})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Processing failed: " + str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
