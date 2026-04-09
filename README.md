# Diabetic Retinopathy Detection and Screening using Deep Learning

## Overview
This project focuses on automated **Diabetic Retinopathy (DR)** analysis using retinal fundus images from the **APTOS 2019 Blindness Detection dataset**.  
The work was developed as a comparative deep learning study to evaluate different problem formulations and model variants for practical DR detection and screening.

The project was not framed as proposing a completely new backbone architecture.  
Instead, the contribution is an **experimentally validated DR screening pipeline** built using transfer learning, attention mechanisms, and systematic comparison of multiple task settings.

---

## Project Objectives
The main objectives of this project were:

- To build an automated system for retinal image based DR detection
- To compare **multiple task formulations** rather than focusing on only one setup
- To evaluate the effectiveness of **EfficientNet-based transfer learning models**
- To study whether adding **CBAM attention** improves lesion-focused feature extraction
- To test ensemble learning as a comparative experiment
- To identify the **most practical formulation for screening use**

---

## Dataset
- **Dataset:** APTOS 2019 Blindness Detection
- **Source:** Kaggle
- **Images:** Retinal fundus images
- **Labels:** DR severity grades from **0 to 4**

### Common dataset structure used
- `train_1.csv`
- `valid.csv`
- `train_images/train_images`
- `val_images/val_images`

---

## Problem Formulations
This project explored three different formulations:

### 1. Five-Class Classification
The original multiclass DR severity classification task:
- `0 = No DR`
- `1 = Mild`
- `2 = Moderate`
- `3 = Severe`
- `4 = Proliferative DR`

This is the most detailed formulation, but also the most challenging because class boundaries are subtle and the dataset is imbalanced.

### 2. Binary Classification
A clinically stricter binary setting:
- `0,1 -> 0`
- `2,3,4 -> 1`

This formulation groups moderate and above DR as positive and is harder than simple screening.

### 3. Screening Binary Classification
A practical screening setup:
- `0 -> 0`
- `1,2,3,4 -> 1`

This setting answers the question:  
**“Is DR present or not?”**

This became the most successful and practically useful formulation in the project.

---

## Preprocessing Pipeline
The stable preprocessing pipeline used in the final work included:

- Reading retinal images using **OpenCV**
- Converting images from **BGR to RGB**
- Cropping black borders
- Resizing images to the required input size
- Converting images to `float32`

### Important note
For the final stable binary screening setup, the preprocessing was intentionally kept simple.  
No heavy handcrafted preprocessing pipeline was used, and in the successful stable version there was **no extra `/255` normalization step** beyond the chosen working pipeline.

---

## Models Used

### EfficientNetB1
EfficientNetB1 was used as a strong transfer learning baseline because it provides a good balance between accuracy and efficiency.

### EfficientNetB0 + CBAM
EfficientNetB0 was combined with **CBAM (Convolutional Block Attention Module)** to improve focus on important retinal regions and lesion-related features.

### Ensemble Learning
An ensemble of model predictions was also tested as a **comparative experiment**.  
However, it should not be claimed as the final best model because the strongest final reported screening result came from **EfficientNetB0 + CBAM**.

---

## Training Strategy
The project used a transfer learning based workflow with:

- Pretrained EfficientNet backbones
- Fine-tuning for retinal image classification
- Class imbalance handling through **class weights**
- Comparative evaluation across task formulations
- Threshold-based binary evaluation at standard threshold `0.5`

Some experiments also explored additional tuning strategies, but only the safe and stable final results are reported here.

### Best Final Practical Model
The **best final practical screening model** in this project is:

**EfficientNetB0 + CBAM on the screening binary split**  
with **98.36% accuracy** and **98.45% F1-score**

---

## Key Findings
Some important observations from the project:

- **5-class classification** is significantly harder because retinal disease stages can look visually similar
- The **harder binary split** improved performance compared to multiclass classification
- The **screening binary split** produced the strongest and most clinically practical results
- **CBAM attention** helped the model focus better on lesion-relevant regions
- Ensemble learning was useful for comparison, but it was **not the final best-performing approach**
- A carefully evaluated and practical screening formulation can be more valuable than forcing a more complex but weaker multiclass claim

---

## Limitations
Some current limitations of the work are:

- Results are based mainly on the **APTOS dataset**
- External validation on another dataset such as **EyePACS** was not included in the final reported pipeline
- The project is best positioned as an **applied comparative study**, not as a claim of architectural novelty

---

## Future Work
Possible future improvements include:

- External validation on additional retinal datasets
- Cross-dataset generalization experiments
- More detailed ablation studies for preprocessing and attention modules

---

## Tech Stack
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Pandas**
- **scikit-learn**
- **Matplotlib / Seaborn** (for analysis and visualization)

---

## Suggested Repository Structure
```bash
├── data/
├── notebooks/
├── models/
├── outputs/
├── app/
├── README.md
```

---

## Conclusion
This project demonstrates that diabetic retinopathy analysis performance depends strongly on the chosen task formulation.  
While five-class severity classification remains challenging, the binary screening formulation produced excellent results and better practical value.

The final outcome of the study shows that **EfficientNetB0 + CBAM** is highly effective for **DR screening on APTOS**, achieving strong accuracy, F1-score, and ROC-AUC while keeping the methodology clear and reproducible.
