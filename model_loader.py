from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-hi-en"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
model.config.tie_word_embeddings = False


def translate_hi_to_en(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    translated = model.generate(
        **inputs,
        max_length=100,
        num_beams=4,          
        early_stopping=True
    )

    return tokenizer.decode(translated[0], skip_special_tokens=True)