import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import evaluate

# 1️⃣ Load model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2️⃣ Define evaluation metrics
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

# 3️⃣ English → French test data
test_data = [
    ("Good morning", "Bonjour"),
    ("How are you?", "Comment ça va ?"),
    ("I love programming.", "J'adore la programmation."),
    ("This is a beautiful day.", "C'est une belle journée."),
    ("Where are you going?", "Où allez-vous ?"),
]

src_lang = "en_XX"
tgt_lang = "fr_XX"

# 4️⃣ Generate translations
preds, refs = [], []
for src, ref in test_data:
    tokenizer.src_lang = src_lang
    inputs = tokenizer(src, return_tensors="pt", truncation=True, padding=True).to(device)

    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        num_beams=5,
        max_new_tokens=128
    )

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    preds.append(translation)
    refs.append(ref)
    print(f"\n🔹 English: {src}\n🔸 Predicted French: {translation}\n✅ Reference: {ref}")

# 5️⃣ Compute metrics
bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
chrf_score = chrf.compute(predictions=preds, references=[[r] for r in refs])

# 6️⃣ Print results
print("\n===== Evaluation Results (English → French) =====")
print(f"BLEU Score : {bleu_score['score']:.2f}")
print(f"chrF Score : {chrf_score['score']:.2f}")
