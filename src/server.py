import io
import base64
import os
from typing import List
import random

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from train import build_model
from predict import load_checkpoint, predict as predict_fn
from xai import GradCAM, get_default_target_layer, overlay_heatmap_on_image

app = FastAPI(title="Potato Leaf Disease API")

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pt")

MODEL_CACHE = {}
DEMO_MODE = False


def get_model(model_path: str = None):
    global DEMO_MODE
    # Use default model path if not specified
    if model_path is None or model_path == "models/best_model.pt":
        model_path = DEFAULT_MODEL_PATH
    
    if model_path in MODEL_CACHE:
        return MODEL_CACHE[model_path]
    
    # Check if model exists
    if not os.path.exists(model_path):
        DEMO_MODE = True
        return None, ["Early_Blight", "Healthy", "Late_Blight"], torch.device('cpu')
    
    DEMO_MODE = False
    model, class_names = load_checkpoint(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    MODEL_CACHE[model_path] = (model, class_names, device)
    return MODEL_CACHE[model_path]


@app.get("/")
async def index():
    # Check if model exists for warning
    model_exists = os.path.exists(DEFAULT_MODEL_PATH)
    warning_html = ""
    if not model_exists:
        warning_html = """
        <div style="background:#fff3cd;border:1px solid #ffc107;padding:12px;border-radius:8px;margin-bottom:16px;">
            <strong>⚠️ Demo Mode:</strong> No trained model found at <code>models/best_model.pt</code>.<br>
            Predictions will be simulated. Train a model first with:<br>
            <code>python run_training.py quick</code>
        </div>
        """
    
    html = f"""
    <html>
    <head>
        <title>🥔 Potato Leaf Disease Classifier</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }}
            h1 {{ color: #2e7d32; }}
            .upload-form {{ background: #f5f5f5; padding: 24px; border-radius: 12px; }}
            input[type="file"] {{ margin: 12px 0; }}
            button {{ background: #4caf50; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; }}
            button:hover {{ background: #388e3c; }}
            label {{ margin-left: 12px; }}
        </style>
    </head>
    <body>
        <h1>🥔 Potato Leaf Disease Classifier</h1>
        {warning_html}
        <div class="upload-form">
            <h3>Upload a potato leaf image</h3>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required /><br>
                <input type="hidden" name="model_path" value="models/best_model.pt" />
                <label><input type="checkbox" name="explain" value="true"/> Show Grad-CAM explanation</label><br><br>
                <button type="submit">🔍 Analyze Leaf</button>
            </form>
        </div>
        <p style="margin-top:20px;color:#666;">
            Classifies potato leaves as: <strong>Early Blight</strong>, <strong>Late Blight</strong>, or <strong>Healthy</strong>
        </p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    model_path: str = Form('models/best_model.pt'),
    explain: str = Form(None),
    as_json: str = Form('false')
):
    global DEMO_MODE
    model, class_names, device = get_model(model_path)
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # prepare tensor (replicate transforms from predict.prepare_image)
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tf_tensor = tf(img).unsqueeze(0)

    # Demo mode - generate fake predictions
    if DEMO_MODE or model is None:
        # Simulate random prediction for demo
        probs = [random.random() for _ in class_names]
        total = sum(probs)
        probs = [p/total for p in probs]
        pred_idx = probs.index(max(probs))
        conf = max(probs)
        prob_vector = probs
        
        result = {
            'prediction': class_names[pred_idx],
            'confidence': float(conf),
            'probabilities': {cls: float(prob_vector[i]) for i, cls in enumerate(class_names)},
            'demo_mode': True
        }
        
        demo_warning = """
        <div style="background:#fff3cd;border:1px solid #ffc107;padding:12px;border-radius:8px;margin-bottom:16px;">
            <strong>⚠️ Demo Mode:</strong> This is a simulated prediction. Train a model for real results!
        </div>
        """
        
        probs_rows = ''.join([f"<tr><td>{cls}</td><td>{result['probabilities'][cls]:.4f}</td></tr>" for cls in class_names])
        html = f"""
        <html>
            <head>
                <title>Prediction Result (Demo)</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 24px; }}
                    table {{ border-collapse: collapse; }}
                    td, th {{ border: 1px solid #ddd; padding: 8px; }}
                    th {{ background: #f6f6f6; }}
                    .card {{ border: 1px solid #eee; padding: 16px; margin-bottom: 16px; border-radius: 8px; max-width: 640px; }}
                    h2 {{ color: #2e7d32; }}
                </style>
            </head>
            <body>
                {demo_warning}
                <div class='card'>
                    <h2>🔍 Prediction (Demo)</h2>
                    <p><b>Class:</b> {result['prediction']}</p>
                    <p><b>Confidence:</b> {result['confidence']:.2%}</p>
                </div>
                <div class='card'>
                    <h2>📊 Probabilities</h2>
                    <table>
                        <tr><th>Class</th><th>Probability</th></tr>
                        {probs_rows}
                    </table>
                </div>
                <div class='card'>
                    <h3>Uploaded Image</h3>
                    <img style='max-width:300px;border-radius:8px;' src='data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}'/>
                </div>
                <p><a href='/'>← Upload another image</a></p>
            </body>
        </html>
        """
        return HTMLResponse(content=html)

    pred_idx, conf, prob_vector = predict_fn(model, tf_tensor, device)
    result = {
        'prediction': class_names[pred_idx],
        'confidence': float(conf),
        'probabilities': {cls: float(prob_vector[i]) for i, cls in enumerate(class_names)}
    }

    use_explain = False
    if isinstance(explain, str):
        use_explain = explain.lower() in ['1','true','yes','on']
    
    cam_img_html = ''
    if use_explain:
        try:
            target = get_default_target_layer(model, arch=model.__class__.__name__.lower())
        except Exception:
            target = get_default_target_layer(model, arch='custom')
        cam = GradCAM(model, target)
        heatmap, _ = cam.generate(tf_tensor.to(device))
        cam.close()
        overlay = overlay_heatmap_on_image(img, heatmap)
        buff = io.BytesIO()
        overlay.save(buff, format='JPEG')
        b64 = base64.b64encode(buff.getvalue()).decode('utf-8')
        result['gradcam_jpeg_base64'] = b64
        cam_img_html = f"<h3>Grad-CAM Explanation</h3><img style='max-width:480px;border:1px solid #ddd;border-radius:8px;' src='data:image/jpeg;base64,{result['gradcam_jpeg_base64']}'/>"

    # If explicitly requested, return pure JSON
    if isinstance(as_json, str) and as_json.lower() in ['1','true','yes','on']:
        return JSONResponse(content=result)

    # Render a friendly HTML page
    probs_rows = ''.join([f"<tr><td>{cls}</td><td>{result['probabilities'][cls]:.4f}</td></tr>" for cls in class_names])

    html = f"""
    <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 24px; }}
                table {{ border-collapse: collapse; }}
                td, th {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background: #f6f6f6; }}
                .card {{ border: 1px solid #eee; padding: 16px; margin-bottom: 16px; border-radius: 8px; max-width: 640px; }}
                h2 {{ color: #2e7d32; }}
                .prediction {{ font-size: 1.5em; color: #1976d2; }}
                .confidence {{ font-size: 1.2em; color: #388e3c; }}
            </style>
        </head>
        <body>
            <div class='card'>
                <h2>🔍 Prediction Result</h2>
                <p class='prediction'><b>Class:</b> {result['prediction'].replace('_', ' ')}</p>
                <p class='confidence'><b>Confidence:</b> {result['confidence']:.2%}</p>
            </div>
            <div class='card'>
                <h2>📊 Probabilities</h2>
                <table>
                    <tr><th>Class</th><th>Probability</th></tr>
                    {probs_rows}
                </table>
            </div>
            <div class='card'>
                <h3>Uploaded Image</h3>
                <img style='max-width:300px;border-radius:8px;' src='data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}'/>
                {cam_img_html}
            </div>
            <p><a href='/'>← Upload another image</a></p>
        </body>
    </html>
    """
    return HTMLResponse(content=html)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
