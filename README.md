# MediBot

A local RAG-based medical chatbot answering questions about cardiovascular and respiratory diseases.

## Getting Started

These instructions will get you a copy of MediBot running on your local machine for development and testing purposes.

### Prerequisites

The things you need before installing the software.

* Python **3.12+**
* Git 
* Ollama
* (Recommended) Virtual environment tool (`venv` or `conda`)

### Installation

1) Clone the repository:
```
git clone https://github.com/vohuutridung/MediBot.git
cd MediBot
```

2) Create & activate a virtual environment (optionally)
```
python -m venv .venv
source .venv/bin/activate # for macOS / Linux
```

3) Install dependencies
```
pip install -r requirements.txt
```

### Model Setup
Before running the app, you must have **Ollama** installed and a local **GGUF** model available to Ollama.

### Environment Variables
Before running **evaluation scripts**, create a `.env` file in the project root with the following:
```
GOOGLE_API_KEY=your_google_api_key_here
```
This key is required for evaluation tasks that rely on Google APIs.  

## Usage

1) Start Ollama
```
ollama serve
```

2) Go to the project and activate your venv
```
cd MediBot
source .venv/bin/activate (in macOS / Linux)
```

3) Run the Streamlit UI
```
streamlit run src/app.py
```

## Additional Documentation and Acknowledgments

* This is a capstone project for Samsung Innovation Campus (SIC) 2025, developed by a four-member team.
* Read [this guide](https://www.gpu-mart.com/blog/import-models-from-huggingface-to-ollama) to see how to import models from Hugging Face into Ollama for local usage.
