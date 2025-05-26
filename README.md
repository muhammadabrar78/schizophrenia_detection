# Schizophrenia EEG Detection Pipeline

This repository provides a complete pipeline for schizophrenia detection using multichannel EEG, following the methodology described in:

> **"Enhanced Schizophrenia Detection Using Multichannel EEG and CAOA-RST-Based Feature Selection"**

## Features

- EEG feature preprocessing and normalization
- Feature extraction using EMD and entropy measures
- Advanced feature selection with CAOA-RST (Crossover-Boosted Archimedes Optimization Algorithm and Rough Set Theory)
- SVM-based classification

## Folder Structure

```
data/
├── features_raw.csv
├── labels.csv

preprocessing.py
feature_extraction.py
feature_selection.py
model.py
main.py
requirements.txt
README.md
```

## Usage

1. **Prepare your data**  
   Place `features_raw.csv` and `labels.csv` in the `data/` folder.

2. **Install requirements**  
   ```
   pip install -r requirements.txt
   ```

3. **Run the pipeline**  
   ```
   python main.py
   ```

## Reference

If you use this code, please cite our paper:

> **"Enhanced Schizophrenia Detection Using Multichannel EEG and CAOA-RST-Based Feature Selection"**

## Contact

For questions or issues, please open an issue on GitHub or contact the corresponding author.

---

**Ready for research, academic, and clinical machine learning projects.**