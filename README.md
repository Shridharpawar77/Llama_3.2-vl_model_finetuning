# Llama_3.2-vl_model_finetuning
A lightweight pipeline for fine-tuning Llama 3.2 Vision on custom image–text datasets. Converts local CSV + images into chat-format messages, applies QLoRA adapters, and trains using TRL’s SFTTrainer for high-quality vision-instruction generation.

# Fine-Tuning Llama 3.2 Vision with Local Image–Text Data
This project demonstrates how to fine-tune **Llama 3.2 Vision** using locally collected images and a CSV-based instruction dataset. It converts each row of your CSV into a chat-style message format, prepares both text and image inputs, and trains the model using **QLoRA** and **TRL’s SFTTrainer** for efficient supervised fine-tuning.

---

## 1. Local Dataset Format
Your CSV must include three columns:
- **image_path** — relative path to each image (joined with `IMAGE_ROOT_DIR`).  
- **instruction** — the user query or task prompt.  
- **description** — the ground-truth response the model should learn.

Each row represents one training sample. Images should be stored locally and accessible through the paths provided in the CSV.

---

## 2. Pipeline Summary
- Load and parse the CSV dataset.  
- Convert each row into a `messages` structure compatible with the Llama 3.2 Vision chat template.  
- Prepare text and image tensors through a custom data collator.  
- Apply **LoRA** adapters (QLoRA configuration) to reduce training cost.  
- Fine-tune the model using TRL’s SFTTrainer with vision-text inputs.

---

## 3. Training
The training script loads the base model in 4-bit precision, attaches LoRA layers, formats each sample through the processor’s chat template, and runs supervised fine-tuning. Checkpoints are stored in your configured output directory.

---

## 4. Notes
- Ensure all image paths in the CSV are valid.  
- Reduce batch size if you encounter memory issues.  
- High-quality instructions and descriptions lead to better model behavior.

---

## 5. Use Case Examples
- Product/image description generation  
- Vision instruction following  
- Custom captioning tasks  
- Domain-specific visual reasoning

---
