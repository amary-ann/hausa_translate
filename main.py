from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# Replace with your HF repo ID, e.g. "username/my-en-ha-translator"
MODEL_NAME = "amyann/my-en-ha-translator"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# FastAPI app
app = FastAPI()

# Request schema
class TranslationRequest(BaseModel):
    text: str

@app.post("/translateHausa")
def translate(req: TranslationRequest):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "success":True,
        "translation": translation
        }
