# Web Frontend Application (Member B - Task B5)

This folder contains the interactive web application for the **Multimodal Fake News Detection** system.

## Overview
The web interface connects to Member A's FastAPI backend running at `http://localhost:8000/predict`.

## Features
1. **Multimodal Input:** Allows users to input news headlines and paste image URLs.
2. **Instant Classification:** Shows `REAL` (green) or `FAKE` (red) with confidence percentage.
3. **Dynamic Gate Weight Gauge:** Visualizes multimodal attention (shows whether the neural network relied on the **headline text** or the **image** to make its decision).
4. **Cross-Modal Consistency Score:** Displays cosine alignment between text semantics and visual features.
5. **Abstention Alert:** Warns the user if the prediction is borderline/ambiguous.

## How to Run
You can open `index.html` directly in your browser, or serve it with any local web server:

```bash
# Option 1: Open directly in your browser
open index.html   # On macOS
start index.html  # On Windows

# Option 2: Run local lightweight server
python -m http.server 3000 --directory web/
```

Then visit: [http://localhost:3000](http://localhost:3000)

## API Data Contract (Contract #4)
* **Endpoint:** `POST http://localhost:8000/predict`
* **Request Payload:**
  ```json
  {
    "text": "Scientists discover water on Mars.",
    "image_url": "https://example.com/mars.jpg"
  }
  ```
* **Response Payload:**
  ```json
  {
    "prediction": "REAL",
    "confidence": 0.9931,
    "label2": 0,
    "gate_weight": 0.5683,
    "cross_modal_similarity": -0.0186,
    "abstained": false,
    "explanations": {
      "text_gate_weight": 0.5683,
      "image_gate_weight": 0.4317,
      "cross_modal_alignment": -0.0186
    }
  }
  ```
