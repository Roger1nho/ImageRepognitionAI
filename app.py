from io import BytesIO

from flask import Flask, request
from flask.cli import load_dotenv
from openai import OpenAI
import openai
import requests
import os
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

load_dotenv()

client = OpenAI(api_key="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")


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(device)

def analyze_with_gpt4v(image_url, prompt):
    """Use OpenAI's GPT-4 Turbo with Vision"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # Updated model name
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"GPT-4 Vision error: {str(e)}")


def analyze_with_clip(image_url, prompt):
    try:
        # Descarcă imaginea
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Definim prompturi comparative
        text_options = [
            prompt,
            "a photo of a dog",
            "a photo of a cat",
            "an outdoor scene",
            "an indoor scene",
            "a black and white photo"
        ]

        # Procesează toate opțiunile simultan
        inputs = clip_processor(
            text=text_options,
            images=image,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        ).to(device)

        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image

        # Normalizare corectă pe toate opțiunile
        probs = logits.softmax(dim=1).squeeze(0)

        # Formatare rezultate
        results = []
        for i, option in enumerate(text_options):
            results.append(f"{option}: {probs[i].item():.2%}")

        return "\n".join(results)

    except Exception as e:
        return f"Error: {str(e)}"

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>OpenAI Image Analyzer</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }}
        form {{ display: flex; flex-direction: column; gap: 10px; }}
        input, textarea, select {{ padding: 8px; }}
        button {{ background: #007bff; color: white; border: none; padding: 10px; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>AI Image Analysis</h1>
    <form method="POST">
        <input type="text" name="image_url" placeholder="Image URL" required>
        <textarea name="prompt" placeholder="Ask something about the image...">What's in this image?</textarea>
        <select name="model">
            <option value="gpt4v">GPT-4 Vision (Detailed)</option>
            <option value="clip">CLIP (Fast Classification)</option>
        </select>
        <button type="submit">Analyze</button>
    </form>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Analysis Result</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }}
        .result {{ background: #f5f5f5; padding: 15px; margin: 20px 0; }}
        a {{ color: #007bff; text-decoration: none; }}
    </style>
</head>
<body>
    <h1>Result ({model_used})</h1>
    <div class="result">{result}</div>
    <a href="/">-- Analyze Another Image</a>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def analyze_image():
    if request.method == "POST":
        return handle_image_analysis(request)
    return show_upload_form()


def show_upload_form():
    """Display the initial form for image analysis"""
    return INDEX_HTML


def handle_image_analysis(request):
    """Process the image analysis request"""
    image_url = request.form["image_url"]
    prompt = request.form.get("prompt", "What's in this image?")
    model_choice = request.form.get("model", "gpt4v")

    result, model_used = get_analysis_result(image_url, prompt, model_choice)
    return render_result(result, model_used)


def get_analysis_result(image_url, prompt, model_choice):
    """Get analysis result from the appropriate model"""
    try:
        if model_choice == "clip":
            result = analyze_with_clip(image_url, prompt)
        else:
            result = analyze_with_gpt4v(image_url, prompt)
        return result, model_choice.upper()
    except Exception as e:
        return f"Error: {str(e)}", "Error"


def render_result(result, model_used):
    """Render the result template"""
    return RESULT_HTML.format(model_used=model_used, result=result)


if __name__ == "__main__":
    print(analyze_with_clip(
        "https://picsum.photos/400/300.jpg",
        "What's in this image?"
    ))
    app.run(debug=True)