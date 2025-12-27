# 🌽 CornCare: Corn Leaf Disease Detection

CornCare is a deep learning project that detects diseases in corn leaves using computer vision.  
It classifies corn leaves into **three categories**:  
- 🌱 **Sehat (Healthy)**  
- 🍂 **Karat (Rust)**  
- 🍃 **Hawar (Blight)**  

Deployed App: [CornCare Streamlit App](https://corncareapp.streamlit.app/)

---

## 📂 Repository Structure

```
cornCare_CV/
│── .gitattributes
│── .gitignore
│── corncare.py           # Streamlit app for deployment
│── best_ResNet.pth       # Trained ResNet50 model
│── best_EfficientNet.pth # Trained EfficientNetB0 model
│── corn_leaf_bg.jpg      # Background image for app
│── main.ipynb            # Training & experimentation notebook
│── requirements.txt      # Dependencies for Streamlit app
```

---


## 📊 Dataset

CornCare is trained on publicly available corn leaf datasets:  
- [Corn or Maize Leaf Disease Dataset (Kaggle - smaranjitghose)](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)  
- [Daun Jagung (Kaggle - taufiqnoviant)](https://www.kaggle.com/datasets/taufiqnoviant/daun-jagung)  
- [Google Drive Mirror](https://drive.google.com/drive/folders/1z0EdlhD1rnSkorFZIfnkqpOI7gdqGwe1?usp=drive_link)

---

## 🚀 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/whoopnes/cornCare_CV.git
   cd cornCare_CV
   ```

2. **Set up environment**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app locally**
   ```bash
   streamlit run corncare.py
   ```

4. **Access the app** at `http://localhost:8501`

---

## 🧠 Model Details

- **ResNet50** : trained as the primary model.  
- **EfficientNet-B0** : tested for performance comparison.  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam (lr=0.001) with `ReduceLROnPlateau` scheduler  
- **Data Augmentation:** Random flips, rotations, and color jitter  

---
