from flask import Flask, request, render_template, jsonify
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import json
import io
from PIL import Image

app = Flask(__name__, template_folder='../templates')

# 1️⃣ Configure the Gemini API Key from environment
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

def pdf_page_to_image(pdf_stream):
    """
    Converts the first page of a PDF file to a PIL Image so we can send it
    to Gemini for multimodal understanding.
    """
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))

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
        # Read PDF and convert 1st page to image
        pdf_bytes = file.read()
        image = pdf_page_to_image(pdf_bytes)

        # 🧠 Choose a supported Gemini model (from Quickstart docs) — e.g., gemini-2.5-flash
        # Supports text + image inputs with structured output capability. :contentReference[oaicite:1]{index=1}
        model = genai.GenerativeModel("gemini-2.5-flash")

        # 🧾 Prompt to extract structured tables
        prompt = """
        Analyze this image. Extract any tables found in it and return them in
        structured JSON. Your final output must be valid JSON with a key
        named "table_data" whose value is a list of row objects.
        """

        # 📤 Call the Gemini API with prompt + image
        response = model.generate_content(
            [prompt, image],
            generation_config={"response_mime_type": "application/json"}
        )

        # Parse the returned JSON
        result = json.loads(response.text)
        return jsonify(result)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Processing failed: " + str(e)}), 500

# Enable debugging locally; Vercel will use app as the entrypoint
if __name__ == "__main__":
    app.run(debug=True)
