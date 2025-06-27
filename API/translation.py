from transformers import MarianMTModel, MarianTokenizer
import torch

# Cihaz kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EN→TR modeli ve tokenizer
en_tr_tokenizer = MarianTokenizer.from_pretrained("translate_models/Helsinki-NLP-en-tr/opus-mt-tc-big-en-tr-tokenizer")
en_tr_model = MarianMTModel.from_pretrained("translate_models/Helsinki-NLP-en-tr/opus-mt-tc-big-en-tr-model").to(device)

# TR→EN modeli ve tokenizer
tr_en_tokenizer = MarianTokenizer.from_pretrained("translate_models/Helsinki-NLP-tr-en/opus-mt-tc-big-tr-en-tokenizer")
tr_en_model = MarianMTModel.from_pretrained("translate_models/Helsinki-NLP-tr-en/opus-mt-tc-big-tr-en-model").to(device)

def translate_tr_to_en(text: str) -> str:
    inputs = tr_en_tokenizer(text, return_tensors="pt", padding=True).to(device)
    output = tr_en_model.generate(**inputs)
    return tr_en_tokenizer.decode(output[0], skip_special_tokens=True)

def translate_en_to_tr(text: str) -> str:
    inputs = en_tr_tokenizer(text, return_tensors="pt", padding=True).to(device)
    output = en_tr_model.generate(**inputs)
    return en_tr_tokenizer.decode(output[0], skip_special_tokens=True)
