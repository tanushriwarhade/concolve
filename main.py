import json
import time
import cv2
import numpy as np
from pdf2image import convert_from_path  # Using pdf2image as alternative to avoid fitz issues
from paddleocr import PaddleOCR
from ultralytics import YOLO
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from fuzzywuzzy import fuzz
import re
import argparse  # For command-line args

# Load models (global for efficiency)
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Switch lang as needed for Hindi/Gujarati ('hi', 'gu')
yolo_model = YOLO('yolov8n.pt')  # Pre-trained; fine-tune on your dataset for signatures/stamps
vlm_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Example master lists (load from actual files in production)
dealer_master = ["ABC Tractors Pvt Ltd", "XYZ Dealers", "Mahindra Dealers"]  # For fuzzy matching
asset_master = ["Mahindra 575 DI", "John Deere 5050D", "Swaraj 855 FE"]  # For exact matching

def ingest_pdf(pdf_path):
    """
    Stage 1: Document Ingestion - Convert PDF to images using pdf2image.
    Requires poppler installed on your system.
    """
    images = convert_from_path(pdf_path)
    return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]

def extract_text_ocr(images):
    """
    Stage 2: Visual and Textual Understanding - Use PaddleOCR for multilingual text extraction.
    Returns list of [bbox, (text, confidence)]
    """
    all_text = []
    for img in images:
        result = ocr.ocr(img, cls=True)
        if result[0] is not None:
            all_text.extend(result[0])
    return all_text

def detect_signature_stamp(images):
    """
    Stage 3: Field Detection - Use YOLO for detecting signatures and stamps.
    Assumes YOLO classes: 0 = signature, 1 = stamp (fine-tune accordingly).
    """
    signature = {"present": False, "bbox": None}
    stamp = {"present": False, "bbox": None}
    for img in images:
        results = yolo_model(img)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                if cls == 0:  # signature
                    signature["present"] = True
                    signature["bbox"] = bbox
                elif cls == 1:  # stamp
                    stamp["present"] = True
                    stamp["bbox"] = bbox
    return signature, stamp

def semantic_reasoning(ocr_results, images):
    """
    Stage 4: Semantic Reasoning - Use VLM (Qwen-VL) to extract and map fields from OCR text.
    Prompt the model with context to identify fields.
    """
    # Combine OCR text for context
    text_context = " ".join([text for _, (text, _) in ocr_results])
    prompt = (
        f"Extract from this invoice text: {text_context}\n"
        "Dealer Name (text, fuzzy match to known dealers)\n"
        "Model Name (text, exact match to known models)\n"
        "Horse Power (numeric, e.g., from '50 HP' extract 50)\n"
        "Asset Cost (numeric, digits only, ignore currency)"
    )
    
    # Process with VLM (using first image for visual context; extend if multi-page)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": images[0]}]}]
    inputs = vlm_processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = vlm_model.generate(**inputs, max_new_tokens=256)
    response = vlm_processor.decode(outputs[0])[0]
    
    # Parse response (adjust regex based on actual output format)
    dealer_match = re.search(r"Dealer Name:\s*(.*)", response)
    model_match = re.search(r"Model Name:\s*(.*)", response)
    hp_match = re.search(r"Horse Power:\s*(\d+)", response)
    cost_match = re.search(r"Asset Cost:\s*(\d+)", response)
    
    dealer = dealer_match.group(1).strip() if dealer_match else ""
    model = model_match.group(1).strip() if model_match else ""
    hp = int(hp_match.group(1)) if hp_match else 0
    cost = int(cost_match.group(1)) if cost_match else 0
    
    return dealer, model, hp, cost

def post_process(dealer, model, hp, cost, ocr_results, signature, stamp):
    """
    Stage 5: Post-Processing - Normalize, match, and compute confidence.
    """
    # Fuzzy match dealer (≥90%)
    best_dealer, score = max([(d, fuzz.ratio(dealer, d)) for d in dealer_master], key=lambda x: x[1], default=("", 0))
    dealer = best_dealer if score >= 90 else ""
    
    # Exact match model
    model = model if model in asset_master else ""
    
    # Confidence: Average OCR conf + detection conf (heuristic)
    ocr_confs = [conf for _, (_, conf) in ocr_results if conf > 0]
    avg_ocr_conf = np.mean(ocr_confs) if ocr_confs else 0.5
    detection_conf = 0.9 if signature["present"] or stamp["present"] else 0.7  # Example from YOLO
    confidence = (avg_ocr_conf * 0.7) + (0.2 if dealer else 0) + (0.1 * detection_conf)
    
    fields = {
        "dealer_name": dealer,
        "model_name": model,
        "horse_power": hp,
        "asset_cost": cost,
        "signature": signature,
        "stamp": stamp
    }
    return fields, confidence

def main(pdf_path, output_path, doc_id="invoice_001"):
    """
    Stage 6: Output Generation - Run the pipeline and save JSON.
    """
    start_time = time.perf_counter()
    
    images = ingest_pdf(pdf_path)
    ocr_results = extract_text_ocr(images)
    signature, stamp = detect_signature_stamp(images)
    dealer, model, hp, cost = semantic_reasoning(ocr_results, images)
    fields, confidence = post_process(dealer, model, hp, cost, ocr_results, signature, stamp)
    
    processing_time = time.perf_counter() - start_time
    cost_estimate = 0.002  # Estimate based on ops (e.g., cloud pricing); adjust
    
    output = {
        "doc_id": doc_id,
        "fields": fields,
        "confidence": round(confidence, 2),
        "processing_time_sec": round(processing_time, 2),
        "cost_estimate_usd": cost_estimate
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    
    print(f"Processed {pdf_path} and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document AI Field Extraction")
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument("--output", default="result.json", help="Path to output JSON")
    parser.add_argument("--doc_id", default="invoice_001", help="Document ID")
    args = parser.parse_args()
    
    main(args.input, args.output, args.doc_id)