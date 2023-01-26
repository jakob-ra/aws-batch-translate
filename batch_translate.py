import ctranslate2

translator = ctranslate2.Translator("opus-mt-de-en",
                                    device="cuda",
                                    device_index=[0],
                                    compute_type='int8_float16')
tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

def batch_translate(texts, batch_size=100):
    """Translate a list of texts in batches."""

    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to('cuda') for k, v in batch.items()}
        translated_batch = translator.translate_batch(batch)
        translated_texts.extend(translated_batch)

    return translated_texts

translate("Hallo Welt")