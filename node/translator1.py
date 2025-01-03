from transformers import MarianMTModel, MarianTokenizer

src_text = [
"good morning.",
]

model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(res[0])