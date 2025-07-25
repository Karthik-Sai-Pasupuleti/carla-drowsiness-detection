

# Drowsiness Detection System

This project aims to detect driver drowsiness using data from a camera and the CARLA simulator. It leverages deep neural networks (DNNs) for feature extraction and a large language model (LLM) to interpret the outputs and control in-car systems like AC, speakers, and tactile feedback on the steering wheel.

---

##  Getting Started

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

##  Project Structure

```
drowsiness-detection/
│
├── .vscode/                   # VSCode settings and configurations
├── src/                       # Source code
│   ├── data_collection/       # Scripts to collect data from camera and CARLA
│   ├── preprocessing/         # Data cleaning, normalization, and augmentation
│   ├── feature_extraction/    # DNN pipelines for feature extraction
│   │   ├── camera_pipeline/   # Camera-based feature extraction
│   │   └── carla_pipeline/    # CARLA simulator-based feature extraction
│   ├── llm_response/          # LLM logic to interpret DNN outputs
│   ├── control_module/        # AC, speaker, and tactile feedback control
│   └── utils/                 # Shared utility functions
│
├── models/                    # Trained or pre-trained models
│   ├── camera_model/
│   └── carla_model/
│
├── data/                      # Dataset storage
│
├── notebooks/                 # Jupyter notebooks for experiments and visualization
│
├── README.md                  # Project overview and setup guide
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files and folders to ignore in Git
```

---
