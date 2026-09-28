# COVID-19 Chest X-ray Classification with Xception

This project classifies **chest X-ray images** (radiographs, not CT scans) into three classes: **COVID-19**, **Normal** and **Viral Pneumonia**. It uses transfer learning from an ImageNet-pretrained Xception network.

This is a course project for the *Artificial Intelligence – Computer Vision* module (November 2025), by **Hadiatou Keita** and **Asmaa Rouchdi**. The notebook and the technical report ([`rapport_technique.pdf`](rapport_technique.pdf)) are in French.

## Data

- **Dataset:** [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (Kaggle, Rahman et al.)
- **Modality:** frontal chest X-rays (PNG)
- **Classes used:** three of the dataset's four image folders. The `Lung_Opacity` folder and the lung masks are not used.

| Class | Images | Train | Validation | Test |
|---|---:|---:|---:|---:|
| COVID | 3,616 | 2,531 | 542 | 543 |
| Normal | 10,192 | 7,134 | 1,529 | 1,529 |
| Viral Pneumonia | 1,345 | 941 | 202 | 202 |
| **Total** | **15,153** | **10,606** | **2,273** | **2,274** |

- **Split:** 70 / 15 / 15 per class (`seed=42`).
- **Class imbalance:** the Normal to Viral Pneumonia ratio is 7.6:1. It is handled with inverse-frequency class weights (COVID 1.40, Normal 0.50, Viral Pneumonia 3.76).

## Method

- **Input:** 299×299 RGB images, Xception `preprocess_input`.
- **Augmentation (training set only):** rotation ±15°, width and height shift 10%, shear 10%, zoom 10%, horizontal flip.
- **Model:** Xception backbone (ImageNet weights, no top) → GlobalAveragePooling → Dense 512 (ReLU, BatchNorm, Dropout 0.5) → Dense 256 (ReLU, BatchNorm, Dropout 0.3) → Dense 3 (softmax). Total: 22.0M parameters.
- **Training in two phases** (Adam, categorical cross-entropy, batch size 32):
  1. The backbone is frozen, lr = 1e-3, up to 15 epochs (1.18M trainable parameters).
  2. The last 30 backbone layers are unfrozen, lr = 1e-5, up to 10 epochs (10.1M trainable parameters).
- **Callbacks:** EarlyStopping (val_loss, patience 5), ReduceLROnPlateau, and ModelCheckpoint on validation accuracy. The best checkpoint (validation accuracy 0.968) was evaluated once on the held-out test set.

## Results (held-out test set, 2,274 images)

These are the outputs saved in the notebook.

| Class | Precision | Recall | F1 | ROC AUC | Support |
|---|---:|---:|---:|---:|---:|
| COVID | 0.959 | 0.948 | 0.954 | 0.995 | 543 |
| Normal | 0.974 | 0.975 | 0.974 | 0.993 | 1,529 |
| Viral Pneumonia | 0.913 | 0.936 | 0.924 | 0.998 | 202 |
| **Macro avg** | 0.949 | 0.953 | 0.951 | | 2,274 |
| **Weighted avg** | 0.965 | 0.965 | 0.965 | | 2,274 |

- **Accuracy:** 96.48%. Micro-averaged ROC AUC: 0.997.
- **Confusion matrix** (rows = true class, columns = predicted: COVID / Normal / Viral Pneumonia):

  | | COVID | Normal | Viral Pneumonia |
  |---|---:|---:|---:|
  | **COVID** | 515 | 27 | 1 |
  | **Normal** | 22 | 1,490 | 17 |
  | **Viral Pneumonia** | 0 | 13 | 189 |

- Most errors confuse COVID with Normal (27 missed COVID cases, 22 false alarms).
- Mean softmax confidence is 0.964 on correct predictions and 0.736 on wrong ones.

## How to run

The notebook was run on Google Colab with a GPU runtime (TensorFlow 2.20.0).

1. Open `COVID19_Classification_Xception.ipynb` in Colab (Runtime → Change runtime type → GPU).
2. Upload your Kaggle API token (`kaggle.json`) when prompted. The notebook downloads and unzips the dataset (about 778 MB).
3. Run all cells. The notebook writes the model, `results.json`, the training history and the classification report to `./results/`. None of these outputs are in the repository.

To run locally instead: `pip install -r requirements.txt`, put `kaggle.json` in the working directory, and start Jupyter.

## Limitations

- **Research exercise, not a medical device.** The model was evaluated on only one public dataset, with no external or clinical validation.
- **The split is per image, not per patient.** The dataset gives no patient identifiers, so images from the same patient may appear in both the training and test sets.
- **Possible source bias.** The classes in this dataset come from different hospitals and online repositories. The model may learn acquisition artefacts (text markers, contrast, borders) instead of pathology. This is a known issue with public COVID-19 X-ray collections (DeGrave et al., *Nature Machine Intelligence*, 2021).
- **No explainability analysis** (for example, Grad-CAM) was performed, so the risk above has not been checked.
- **Only one run** (one seed), so there are no confidence intervals.

## Possible improvements

- Evaluate on an external chest X-ray dataset from a different source.
- Add Grad-CAM saliency maps and check whether the model looks at the lungs.
- Include the `Lung_Opacity` class, or use the provided lung masks to crop out non-lung regions.
- Repeat training with several seeds and report the variance.

## Authors

- Hadiatou Keita: [@hadiatou4](https://github.com/hadiatou4)
- Asmaa Rouchdi

## References

1. F. Chollet, *Xception: Deep Learning with Depthwise Separable Convolutions*, CVPR 2017.
2. T. Rahman, M. Chowdhury et al., COVID-19 Radiography Database, Kaggle.
3. A. J. DeGrave, J. D. Janizek, S.-I. Lee, *AI for radiographic COVID-19 detection selects shortcuts over signal*, Nature Machine Intelligence, 2021.

> Educational project. Not intended for clinical use.
