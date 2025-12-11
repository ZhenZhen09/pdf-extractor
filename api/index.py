from flask import Flask, request, render_template, jsonify
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import json
import io
from PIL import Image

app = Flask(__name__, template_folder='../templates')

# Configure Google Gemini API Key from environment variables
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

def pdf_page_to_image(pdf_stream):
    """
    Converts the first page of a PDF to a PIL Image object.
    Gemini can process PIL Images directly.
    """
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    page = doc.load_page(0)  # Get first page
    pix = page.get_pixmap()   # Render page to image

    # Convert PyMuPDF pixmap to PIL Image
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
        # Convert PDF first page to image
        file_bytes = file.read()
        image = pdf_page_to_image(file_bytes)

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5')  # <-- use supported model

        # Prompt for table extraction
        prompt = """
        Analyze this image. Extract any tabular data found into a structured JSON format. 
        The JSON must contain a key named 'table_data' which is an array of objects representing the rows.
        Ensure every row object has consistent keys based on the table headers.
        """

        # Generate content from Gemini
        response = model.generate_content(
            [prompt, image],
            generation_config={"response_mime_type": "application/json"}
        )

        # Parse JSON response
        data = json.loads(response.text)

        return jsonify(data)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

# Required for local testing; Vercel will use app as entrypoint
if __name__ == '__main__':
    app.run(debug=True)
