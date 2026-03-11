import os
import json
import logging
import time
import base64
import re
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from openai import OpenAI

QUALITY_EVAL_PROMPT = """
**Role:** You are a meticulous Image Quality Analyst.

**Objective:** Your task is to evaluate the **overall visual fidelity** and technical quality of the **batch of {num_frames} image frames** provided. These frames are sampled from a single video clip.

**CRITICAL RULES:**
1. **Ignore Artistic Blur:** Do NOT penalize background bokeh/depth-of-field.
2. **Ignore Occlusion:** Do NOT penalize if the subject is partially blocked.

---

### Core Evaluation Criteria
Critically assess these aspects across all provided frames to determine your **average score**.

1.  **Subject Sharpness & Clarity:**
    * Are the **visible portions** of the **Main Subject** crisp and defined (on average)?
    * Are fine details preserved?
    * Are the frames free from global "softness" or low-resolution haziness?

2.  **Artifacts & Distortion:**
    -   **AI Artifacts:** Are there "melting" textures, distorted faces/hands?
    -   **Compression:** Are there visible blocks, banding, or ringing artifacts?
    -   **Noise:** Is there unintended grain that degrades the details?

3.  **Lighting & Visual Integrity:**
    -   Is the exposure balanced (subject is visible)?
    -   Are colors natural and consistent?

---

### EVALUATION RUBRIC (Strict 1-3 Scale)

- **1 (Reject / Unusable):**
  **Severe Technical Failure.** The main subject is unrecognizable, heavily blurred (technical blur), or suffers from gross AI distortions (melted faces/limbs). The image is broken.

- **2 (Passable / Average):**
  **Noticeable Imperfections.** The subject is structurally correct but lacks fine detail. Looks "soft," "waxy," or has visible noise/artifacts. Usable, but clearly digital or low-res.

- **3 (Excellent / High Quality):**
  **Professional Standard.** The main subject is **razor-sharp** with rich textures (hair/skin visible). No visible noise, compression, or AI artifacts. Looks like high-end photography.

---

### Output Format
Return a single JSON object with the integer score (1, 2, or 3).

**Example:**
{{
  "score": 3,
  "justification": "Subject is razor-sharp. No artifacts."
}}
"""

def extract_frames_from_video(video_path, num_sample=6):
    base64_images = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1: return []
        
        indices = np.linspace(0, total_frames - 1, num_sample).astype(int)
        if len(indices) >= 4:
            indices = indices[1:-1]
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count in indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                base64_images.append(f"data:image/jpeg;base64,{img_str}")

            frame_count += 1

    except Exception as e:
        print(f"Error extracting frames: {e}")
    finally:
        cap.release()
    return base64_images

def parse_response(content):
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        json_str = match.group(0) if match else content
        data = json.loads(json_str)
        return int(data.get("score", 0)), data.get("justification", "")
    except:
        return "null", content

def evaluate_single_video(video_path, client, model_name):
    frames = extract_frames_from_video(video_path)
    if not frames:
        return "null", "Frame extraction failed"

    prompt = QUALITY_EVAL_PROMPT.format(num_frames=len(frames))
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Evaluate these frames."}
        ] + [{"type": "image_url", "image_url": {"url": f}} for f in frames]}
    ]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0
        )
        content = completion.choices[0].message.content
        return parse_response(content)
    except Exception as e:
        print(f"GPT Error: {e}")
        return "null", str(e)

def eval_image_quality(data_item, result_json_path, gpt_url, gpt_key, gpt_model):

    client = OpenAI(base_url=gpt_url, api_key=gpt_key)
    
    task_id = data_item.get('task_id')
    video_path = data_item.get('video_path')
    
    if not video_path or not os.path.exists(video_path):
        print(f"  [ImageQuality] Video not found: {video_path}")
        return 0

    score, justification = evaluate_single_video(video_path, client, gpt_model)
    
    new_record = {
        "task_id": task_id,
        "video_path": video_path,
        "score": score,
        "justification": justification
    }
    
    if isinstance(score, int):
        try:
            with open(f"{result_json_path}/{data_item['task_id']}.json", 'w', encoding='utf-8') as f:
                json.dump(new_record, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"  [ImageQuality] Error saving result: {e}")

    return score
