🌍 mBART-50 Multilingual Translator

A multilingual text translation web application built using Streamlit, PyTorch, and Hugging Face Transformers, powered by Facebook AI’s mBART-50 model.

This application supports translation between 50+ languages in a simple and interactive web interface.

🚀 Live Demo

🔗 Deployed App Link:
👉(https://multilanguage-translator-cxwgk69wc3qcfjdedvh2eq.streamlit.app/)
📌 Features

🌐 Supports 50+ languages

🔁 Many-to-many translation (any language → any language)

⚡ Beam search for improved translation quality

🖥️ Clean and interactive UI using Streamlit

📊 Evaluation using BLEU and chrF metrics

📈 Visualization of evaluation scores

🧠 Model Details

Model Name: facebook/mbart-large-50-many-to-many-mmt

Framework: PyTorch

Tokenizer: MBart50TokenizerFast

Source: Hugging Face Transformers

The model is a sequence-to-sequence multilingual model capable of translating between multiple languages without pivoting through English.

📂 Project Structure
├── test.py                    # Streamlit web app
├── evaluate_translations.py   # BLEU & chrF evaluation script
├── plot_evaluation.py         # Evaluation metrics visualization
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
⚙️ Installation (Run Locally)
1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run the Application
streamlit run test.py

Then open the local URL shown in your terminal (usually http://localhost:8501).

📊 Evaluation Metrics

The model was evaluated using:

BLEU Score

chrF Score

Example Results:

Metric	Score
BLEU	31.03
chrF	66.04

You can run evaluation using:

python evaluate_translations.py

To generate evaluation graph:

python plot_evaluation.py
🌎 Supported Languages (Sample)

English

Hindi

French

Spanish

German

Arabic

Chinese

Japanese

Bengali

Tamil
…and many more (50+ total).

🛠️ Tech Stack

Python

Streamlit

PyTorch

Transformers (Hugging Face)

Matplotlib

⚠️ Notes

The model size is large (~2GB), so first load may take time.

GPU is recommended for faster inference.

Free deployment platforms may have memory limits.

🎯 Use Cases

Language learning

Academic projects

NLP research demonstrations

Portfolio showcase

Multilingual applications prototype

📌 Future Improvements

Add text-to-speech

Add speech-to-text

Optimize model for faster inference

Deploy using Docker + Cloud GPU

Add translation history feature
