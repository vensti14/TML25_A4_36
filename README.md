

# Explainability Project: Network Dissection, Grad-CAM, and LIME

> **Interpreting Deep Learning Models via Neuron Dissection and Visual Explanations**

---

## 📝 Overview

This project investigates the inner workings and decision-making processes of deep neural networks using three leading explainability techniques:

* **Network Dissection:** Maps neurons to human-understandable concepts.
* **Grad-CAM (and variants):** Generates heatmaps showing model “attention.”
* **LIME:** Provides superpixel-level, model-agnostic visual explanations.

The project includes systematic analysis and comparison of these methods on standard datasets and pretrained models (ResNet18/ResNet50).

---

## 📂 Repository Structure

```
.
├── CLIP-dissect/             # Network Dissection code and results
├── gradcam_explanations/     # Scripts and visualizations for Grad-CAM, ScoreCAM, AblationCAM
├── lime_explanations/        # LIME scripts, parameter search, masks, and overlays
├── images/                   # Test images (ImageNet subset, etc.)
├── reports/                  # PDF/Markdown reports and figures
├── lime_parameters.pkl       # Submission file: LIME parameters dictionary
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

---

## 🚀 Tasks & Methods

### **Task 1: Network Dissection**

* Labels neurons in the last 3 layers of ResNet18 (ImageNet & Places365) with semantic concepts using the CLIP-dissect toolkit.
* Visualizes which concepts are most prevalent, and compares the two models.

### **Task 2: Grad-CAM and Variants**

* Applies Grad-CAM, AblationCAM, and ScoreCAM to highlight salient image regions for model predictions.
* Saves and compares the resulting heatmaps.

### **Task 3: LIME Explanations**

* Uses LIME (Local Interpretable Model-agnostic Explanations) to create superpixel-based masks for each image.
* Tunes parameters for maximum interpretability and leaderboard IOU.

### **Task 4: Comparison & Report**

* Computes Intersection over Union (IoU) to compare regions highlighted by Grad-CAM and LIME.
* Summarizes findings in detailed reports.

---

## 📊 Results Summary

* **Network Dissection**: Revealed object-centric vs. scene-centric concepts across models.
* **Grad-CAM**: Consistently identified major objects but with broad, sometimes diffuse heatmaps.
* **LIME**: Provided more localized, sometimes fragmented, but interpretable explanations.
* **Comparison**: Agreement (IoU) between methods was higher for simple images, lower for complex scenes.

See `reports/` for full analysis, figures, and detailed insights.

---

## 🛠️ Getting Started

### **Requirements**

* Python 3.8+
* See `requirements.txt` for package dependencies (PyTorch, torchvision, lime, scikit-image, matplotlib, etc.)

### **Setup**

```bash
pip install -r requirements.txt
```

### **Example Usage**

#### **Network Dissection**

```bash
cd CLIP-dissect
python describe_neurons.py --clip_model RN50 --target_model resnet18 --target_layers layer3,layer4,fc --d_probe imagenet_val --result_dir ./results_imagenet
```

#### **Grad-CAM**

```python
# See gradcam_explanations/ for scripts to generate Grad-CAM, ScoreCAM, and AblationCAM overlays.
```

#### **LIME**

```python
# See lime_explanations/ for LIME explainer scripts and parameter tuning.
```

#### **LIME Parameter Submission**

```python
import pickle
with open('lime_parameters.pkl', 'rb') as f:
    params = pickle.load(f)
```

---

## 📸 Example Visualizations

<p align="center">
  <img src="reports/gradcam_goldfish.png" width="300"/>
  <img src="reports/lime_goldfish.png" width="300"/>
</p>
<sub>Left: Grad-CAM heatmap for "goldfish". Right: LIME mask for the same image.</sub>

---



## 📄 License

This project is for educational and academic use only.
Check individual tool licenses for more details.

---


---


