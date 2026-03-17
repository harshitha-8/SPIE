---
title: QuantumHarvest Cotton Defoliation Intelligence
emoji: 🌿
colorFrom: violet
colorTo: cyan
sdk: gradio
sdk_version: 5.16.2
app_file: app.py
pinned: false
---

# 🌿 QuantumHarvest: Cotton Defoliation Intelligence

[![GitHub Repository](https://img.shields.io/badge/GitHub-View_Repository-blue?logo=github)](https://github.com/harshitha-8/SPIE)
[![Gradio Space](https://img.shields.io/badge/Gradio-Web_App-ff6600?logo=gradio)](https://huggingface.co/spaces/Harshitha09/quantum_Harvest)

QuantumHarvest is a **Hybrid Quantum-Classical Machine Learning** application designed to evaluate the defoliation readiness of cotton fields using UAV (drone) RGB imagery. 

By utilizing advanced quantum variational circuits (VQC) and computer vision, this tool accurately classifies fields into **Pre-Defoliation** or **Post-Defoliation** stages, alongside a robust algorithm to physically count visible cotton bolls.

For full research details, data access, and codebase, please visit the main project repository: **[harshitha-8/SPIE](https://github.com/harshitha-8/SPIE)**.

---

## ✨ Key Features

1. **Hybrid Classification Engine**
   - Extracts 12 spectral and textural (GLCM) features classically.
   - Leverages a **Qiskit `ZZFeatureMap` and `RealAmplitudes` Ansatz** to rank and select the 4 most robust, noise-agnostic features (e.g., Green Variability, Red-Blue Ratio).
   - Achieves near-perfect accuracy using a fine-tuned RBF Support Vector Machine (SVM) on the quantum-selected feature subset.

2. **Advanced Boll Detection & Counting**
   - Uses multi-scale Top-Hat transformations and CLAHE (Contrast Limited Adaptive Histogram Equalization) to aggressively detect cotton bolls.
   - Dynamically handles the severe camouflage of pre-defoliation bolls (hidden under green canopies and shadows) versus fully exposed post-defoliation bolls.
   - Outputs an annotated image plotting every localized boll.

3. **Interactive Gradio Workspace**
   - A highly customized, cyberpunk-inspired dark theme UI.
   - Real-time generation of confidence metrics, feature distribution bar charts, and quantum subset rankings.

## 🚀 How to Run Locally

You will need Python 3.10+ (or newer). We recommend using a virtual environment.

### 1. Clone the Repository
```bash
git clone https://github.com/harshitha-8/SPIE.git
cd SPIE/CottonDefoliationApp # Or your local path to this UI code
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: Make sure you have `qiskit`, `opencv-python-headless`, `gradio`, and `scikit-learn` installed.*

### 3. Run the Application
```bash
python app.py
```
Once the server starts, open your browser and navigate to the local address provided (usually `http://127.0.0.1:7860`).

On Hugging Face Spaces, the app now uses the platform-provided `PORT` automatically and binds to `0.0.0.0`, which is required for hosted deployment.

## Deploy To Hugging Face Space

Hugging Face Spaces uses its own git repository. Updating this GitHub repository alone does not update the live Space app.

To publish this exact app to your Space, push the same files to the Space repo:

```bash
git clone https://github.com/harshitha-8/SPIE.git
cd SPIE
git remote add hf https://huggingface.co/spaces/Harshitha09/quantum_Harvest
git push hf main
```

If `hf` already exists as a remote, update it instead:

```bash
git remote set-url hf https://huggingface.co/spaces/Harshitha09/quantum_Harvest
git push hf main
```

You will be prompted for your Hugging Face credentials or access token with write permission.

---

## 📸 Using the Application

1. **Upload an Image**: Click the "📂 CHOOSE IMAGE FILE" button and select an aerial RGB drone photo of a cotton field.
2. **Analyze**: Click "⚛ RUN QUANTUM ANALYSIS".
3. **Review Results**:
   - **Verdict**: Immediate confirmation of Pre/Post Defoliation status.
   - **Boll Count**: Estimated count of visible or camouflaged cotton bolls.
   - **Detection Map**: Download or enlarge the annotated image showing the physical detection hit-boxes.
   - **Metrics**: Review the Quantum Subset charts and the confidence spectrum.

## 🧠 Model Training

If you are using new data and wish to retrain the underlying SVM model:
1. Prepare your extracted features using the methods laid out in the main [SPIE repository](https://github.com/harshitha-8/SPIE).
2. Run `python train_model.py` to regenerate the `model.pkl` file used by the application.

## 🤝 Citation & Credits
If you use this codebase or application for research, please reference the main repository: [https://github.com/harshitha-8/SPIE](https://github.com/harshitha-8/SPIE).

*Built with ♥ using Qiskit, Gradio, OpenCV, and Scikit-Learn.*
