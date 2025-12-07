import os
import base64
import json
from typing import List, Dict

import pandas as pd
from openai import OpenAI

# =========================
# CONFIG
# =========================
# vLLM server (OpenAI compatible)
VLLM_BASE_URL = "http://localhost:8000/v1"   # change if different
VLLM_API_KEY = "EMPTY"                       # ignored by vLLM

MODEL_NAME = "llama32-vision-merged"        # merged fine-tuned model name/path as used in vllm serve

# Paths
IMAGE_DIR = "./images"                       # folder with your images
OUTPUT_TEXT_DIR = "./image_descriptions"    # where to save .txt files
OUTPUT_EXCEL = "./image_analysis.xlsx"      # final Excel file

# Prompt for the model
SYSTEM_PROMPT = (
    "You are a careful, neutral image analyst. "
    "For each image, you must:\n"
    "1) Write a short, clear description of what is visible.\n"
    "2) Decide whether the *environment* (not the people themselves) strongly suggests severe poverty, "
    "slum-like informal settlements, or very poor living conditions.\n\n"
    "Return ONLY a valid JSON object with two keys:\n"
    "  - \"description\": string\n"
    "  - \"settlement_decision\": string, either \"settlement_case\" or \"normal_case\".\n\n"
    "Use \"settlement_case\" only if the surroundings clearly indicate severe deprivation or informal settlements; "
    "otherwise use \"normal_case\"."
)

USER_INSTRUCTION = (
    "Look at this image, describe it briefly, and decide if it should be labeled as a settlement case "
    "based on the environment."
)

# Allowed image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# =========================
# HELPER FUNCTIONS
# =========================

def get_client() -> OpenAI:
    """Create an OpenAI client pointing to vLLM."""
    client = OpenAI(
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
    )
    return client


def encode_image_to_base64(path: str) -> str:
    """Read image from disk and return base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def list_image_files(folder: str) -> List[str]:
    """Return full paths of all image files under a folder."""
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, name))
    files.sort()
    return files


def query_model_for_image(
    client: OpenAI,
    model_name: str,
    image_path: str,
    user_instruction: str,
) -> Dict[str, str]:
    """
    Send one image to the model and parse JSON with:
      - description
      - settlement_decision
    """
    image_b64 = encode_image_to_base64(image_path)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                ],
            },
        ],
        max_tokens=256,
        temperature=0.2,
    )

    # vLLM / OpenAI-style response
    message = response.choices[0].message

    # Some servers return a list of content blocks; handle both cases
    if isinstance(message.content, list):
        full_text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
    else:
        full_text = message.content

    # Safe JSON parse with fallback
    try:
        data = json.loads(full_text)
        description = data.get("description", "").strip()
        decision = data.get("settlement_decision", "").strip()
    except json.JSONDecodeError:
        # Fallback: if model didn't obey, use raw text and default decision
        description = full_text.strip()
        decision = "normal_case"

    if decision not in {"settlement_case", "normal_case"}:
        decision = "normal_case"

    return {
        "description": description,
        "settlement_decision": decision,
    }


def save_description_text(
    output_dir: str,
    image_name: str,
    description: str,
    settlement_decision: str,
) -> None:
    """Save a .txt file per image with description + decision."""
    os.makedirs(output_dir, exist_ok=True)
    base_name, _ = os.path.splitext(image_name)
    txt_path = os.path.join(output_dir, f"{base_name}.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Image: {image_name}\n")
        f.write(f"Settlement decision: {settlement_decision}\n\n")
        f.write("Description:\n")
        f.write(description)

    # Optional: print path for debug
    # print(f"Saved: {txt_path}")


def build_excel(
    rows: List[Dict[str, str]],
    excel_path: str,
) -> None:
    """
    Create an Excel file with columns:
      - image_name
      - image_description
      - settlement_decision
    """
    df = pd.DataFrame(rows, columns=["image_name", "image_description", "settlement_decision"])
    df.to_excel(excel_path, index=False)
    print(f"Excel saved to: {excel_path}")


# =========================
# MAIN PIPELINE
# =========================

def run_batch_inference():
    client = get_client()
    image_paths = list_image_files(IMAGE_DIR)
    print(f"Found {len(image_paths)} images in {IMAGE_DIR}")

    results = []

    for img_path in image_paths:
        image_name = os.path.basename(img_path)
        print(f"Processing: {image_name}")

        model_output = query_model_for_image(
            client=client,
            model_name=MODEL_NAME,
            image_path=img_path,
            user_instruction=USER_INSTRUCTION,
        )

        description = model_output["description"]
        decision = model_output["settlement_decision"]

        # Save per-image text file
        save_description_text(
            output_dir=OUTPUT_TEXT_DIR,
            image_name=image_name,
            description=description,
            settlement_decision=decision,
        )

        # Collect for Excel
        results.append(
            {
                "image_name": image_name,
                "image_description": description,
                "settlement_decision": decision,
            }
        )

    # Build Excel file
    build_excel(results, OUTPUT_EXCEL)


if __name__ == "__main__":
    run_batch_inference()
