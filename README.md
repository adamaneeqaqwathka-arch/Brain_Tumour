# Brain Tumor MRI Classification

Deep-learning classifier that sorts a brain MRI slice into **glioma, meningioma, pituitary tumour or no tumour**, with a Streamlit app that shows the prediction, the confidence for every class and a Grad-CAM heat-map.

> **Educational prototype - not a medical device.** See "Limitations" before drawing any conclusion from the numbers.

## Results (held-out test set, 246 images)

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| Custom CNN, baseline (**chosen**) | 0.809 (95% interval 0.756-0.854) | 0.801 |
| Custom CNN, tuned | 0.752 | 0.743 |
| Custom CNN, head-cropped input | 0.756 | 0.746 |
| *Reference:* random forest on 6 global image statistics (no CNN) | 0.854 | - |
| MobileNetV2 / EfficientNet-B0 / ResNet50 (transfer learning) | **not run yet** - see below | |

The model is chosen by validation macro-F1 first; the test set is used once for the numbers above.

## Limitations (important)
* **Shortcut learning.** The four classes were scanned in different planes and sequences, so brightness, framing and sharpness alone predict the class (the random forest above). Grad-CAM shows the chosen CNN often looks at the black background and skull edge instead of the tumour. A head-cropped, brightness-normalised input moves the model's attention onto the head (74% vs 49% of the Grad-CAM weight) without raising accuracy.
* **Transfer learning is implemented but not yet run.** The environment used to build this project could not download the pretrained ImageNet weights. The code for MobileNetV2, EfficientNet-B0 and ResNet50 (new head, then fine-tuning with a learning-rate search) is in the notebook and was tested for errors only. Run it on a machine with internet or a GPU (e.g. Google Colab) to produce those results; the model-selection cell then picks the best model automatically.
* One public dataset, 2,443 slices, small test set. Not validated on other scanners or hospitals.

## Repository layout
```
notebooks/Brain_Tumor_MRI_Classification.ipynb   full project notebook (EDA, preprocessing, models, comparison, Grad-CAM, save/load)
app/streamlit_app.py                              the web app
app/model_utils.py                                model loading, pre-processing, prediction, Grad-CAM
models/best_model.pt, model_info.json             chosen model and its settings/test metrics
sample_images/                                    8 test scans for trying the app
requirements.txt
data/                                             NOT in git - see below
```

## Data layout
Download the Brain Tumor MRI multi-class dataset and arrange it as:
```
data/train/<glioma|meningioma|no_tumor|pituitary>/*.jpg
data/valid/<same four folders>/
data/test/<same four folders>/
```
(2,443 images: train 1,695 / valid 502 / test 246, all 640x640.) On Colab the first notebook cell builds this layout from uploaded zip files.

## Run the app
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
Upload a JPG/PNG scan or pick a sample. A warning appears when the top confidence is below 60%.

## Run the notebook
* **Google Colab (recommended):** upload the notebook, choose *Runtime > Change runtime type > T4 GPU*, upload the dataset zip files to the Files panel, *Runtime > Run all* (roughly 20-40 minutes on a T4; not yet timed).
* **Locally:** place the data in `data/`, install the requirements, run all cells. On a CPU the custom CNN takes about 1 hour and the transfer models several more.
* If the machine has no internet, put the three torchvision ImageNet files (`mobilenet_v2-*.pth`, `efficientnet_b0_*.pth`, `resnet50-*.pth`) in a folder named `pretrained/`.

## Method summary
Images resized to 256x256 (chosen from a detail-vs-compute study), ImageNet standardisation, training-only augmentation (horizontal flip, rotation 15 deg, shift, zoom, brightness/contrast jitter; no vertical flip), class-weighted loss, AdamW, ReduceLROnPlateau, early stopping and best-weight checkpointing. Metrics: accuracy, macro precision/recall/F1, confusion matrices, bootstrap interval on the test set.
