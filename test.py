import streamlit as st
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def load_model_and_tokenizer(model_name="facebook/mbart-large-50-many-to-many-mmt"):
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

@st.cache_resource
def get_resources():
    return load_model_and_tokenizer()

MODEL, TOKENIZER, DEVICE = get_resources()

LANGUAGES = {
    "Arabic": "ar_AR",
    "Czech": "cs_CZ",
    "German": "de_DE",
    "English": "en_XX",
    "Spanish": "es_XX",
    "Estonian": "et_EE",
    "Finnish": "fi_FI",
    "French": "fr_XX",
    "Gujarati": "gu_IN",
    "Hindi": "hi_IN",
    "Italian": "it_IT",
    "Japanese": "ja_XX",
    "Kazakh": "kk_KZ",
    "Korean": "ko_KR",
    "Lithuanian": "lt_LT",
    "Latvian": "lv_LV",
    "Burmese": "my_MM",
    "Nepali": "ne_NP",
    "Dutch": "nl_XX",
    "Romanian": "ro_RO",
    "Russian": "ru_RU",
    "Sinhala": "si_LK",
    "Turkish": "tr_TR",
    "Vietnamese": "vi_VN",
    "Chinese": "zh_CN",
    "Afrikaans": "af_ZA",
    "Azerbaijani": "az_AZ",
    "Bengali": "bn_IN",
    "Persian": "fa_IR",
    "Hebrew": "he_IL",
    "Croatian": "hr_HR",
    "Indonesian": "id_ID",
    "Georgian": "ka_GE",
    "Khmer": "km_KH",
    "Macedonian": "mk_MK",
    "Malayalam": "ml_IN",
    "Mongolian": "mn_MN",
    "Marathi": "mr_IN",
    "Polish": "pl_PL",
    "Pashto": "ps_AF",
    "Portuguese": "pt_XX",
    "Swedish": "sv_SE",
    "Swahili": "sw_KE",
    "Tamil": "ta_IN",
    "Telugu": "te_IN",
    "Thai": "th_TH",
    "Tagalog": "tl_XX",
    "Ukrainian": "uk_UA",
    "Urdu": "ur_PK",
    "Xhosa": "xh_ZA",
    "Galician": "gl_ES",
    "Slovene": "sl_SI",
}

st.title("mBART-50 Multilingual Translator")
source_name = st.selectbox("Source language", sorted(LANGUAGES.keys()), index=sorted(LANGUAGES.keys()).index("English") if "English" in LANGUAGES else 0)
target_name = st.selectbox("Target language", sorted(LANGUAGES.keys()), index=sorted(LANGUAGES.keys()).index("Hindi") if "Hindi" in LANGUAGES else 1)
source_code = LANGUAGES[source_name]
target_code = LANGUAGES[target_name]

text = st.text_area("Enter text to translate", height=200)
if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter some text to translate.")
    elif source_code == target_code:
        st.info("Source and target languages are the same. Showing original text.")
        st.write(text)
    else:
        TOKENIZER.src_lang = source_code
        inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        forced_bos_token_id = TOKENIZER.lang_code_to_id[target_code]
        generated_tokens = MODEL.generate(**inputs, forced_bos_token_id=forced_bos_token_id, num_beams=5, max_new_tokens=512)
        translation = TOKENIZER.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        st.subheader("Translation")
        st.write(translation)

st.markdown("---")
st.caption("Model: facebook/mbart-large-50-many-to-many-mmt | Tokenizer: MBart50TokenizerFast")
