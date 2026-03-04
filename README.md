# ENSF617 Garbage Classifier

This project explores garbage-sorting classification using three different setups: image-only, text-only, and multimodal. The goal is to predict one of four disposal classes, `Black`, `Blue`, `Green`, or `TTR`/Other using a dataset crowdsourced in the ENSF617 course.

We compare:

- an image model built on EfficientNet-V2-M,
- a text model built on DistilBERT using text extracted from image filenames,
- and a multimodal model that combines both.

## Problem Statement

The task is to classify each waste item into the correct disposal bin using either:

- the item image,
- a text description derived from the filename,
- or both together.

This repository evaluates all three approaches under the same training and testing pipeline.

## Method

### Model Design

- **Image model**: pretrained EfficientNet-V2-M model
- **Text model**: DistilBERT
- **Multimodal model**: concatenates image and text feature vectors

### Data Representation

- Images are loaded using `ImageFolder`
- Text input is created directly from the filename by lowercasing and replacing underscores with spaces
- Training augmentation includes random rotation and random horizontal flip
- Validation and test images use the EfficientNet inference transform

## Training/Testing Outputs

For each model, the main saved outputs are:

- `metrics.json`
- `confusion_matrix.png`
- `roc_curves.png`
- `misclassified.csv`

These are stored in:

- `src/artifacts/best_image/test/`
- `src/artifacts/best_text/test/`
- `src/artifacts/best_multimodal/test/`

The notebook `artifact_results.ipynb` compares these outputs side by side.

## Results

### Overall Test Performance

| Model      | Accuracy | Macro F1 | Macro ROC AUC |  Loss | Errors |
| ---------- | -------: | -------: | ------------: | ----: | -----: |
| Image      |    0.738 |    0.727 |         0.918 | 0.667 |    899 |
| Text       |    0.793 |    0.784 |         0.913 | 0.812 |    709 |
| Multimodal |    0.757 |    0.750 |         0.931 | 0.674 |    833 |

### Per-Class F1

| Class | Image |  Text | Multimodal |
| ----- | ----: | ----: | ---------: |
| Black | 0.565 | 0.692 |      0.646 |
| Blue  | 0.748 | 0.818 |      0.778 |
| Green | 0.864 | 0.854 |      0.854 |
| TTR   | 0.730 | 0.772 |      0.722 |

## Discussion

One interesting result is that the text-only model performs best on standard classification metrics, while the multimodal model achieves the highest macro ROC AUC. This seems to be largely because the filenames are often very informative: many of them already describe the object or material clearly. By comparison, the image-only model has a harder time separating items that look visually similar across disposal categories, especially with a frozen image encoder.

Looking more closely at the overall numbers, the text model improves accuracy from `0.738` to `0.793` compared with the image model, and improves macro F1 from `0.727` to `0.784`. It also reduces the total number of test errors from `899` to `709`, which is a drop of `190` mistakes. The multimodal model sits between the two on top metrics with `0.757` accuracy and `0.750` macro F1, but it has the best macro ROC AUC at `0.931`, higher than both the image model (`0.918`) and text model (`0.913`).

The image-only model performs the worst of the three. That makes sense for this dataset: items from `Black`, `Blue`, and `TTR` can look quite similar, especially when packaging shape or appearance is inconsistent. Keeping the image encoder frozen also limits how much the model can adapt to this specific task.

The per-class numbers show the same pattern. For `Black`, F1 rises from `0.565` in the image model to `0.692` in the text model, while the multimodal model reaches `0.646`. For `Blue`, the text model is strongest with F1 `0.818`, compared with `0.748` for image and `0.778` for multimodal. `Green` is strong in all three settings, staying around `0.854` to `0.864`, which suggests this class is comparatively easy to separate. `TTR` is more difficult: the text model gets the best F1 at `0.772`, the image model follows at `0.730`, and the multimodal model drops slightly to `0.722`.

The multimodal model does not outperform the text model on accuracy or macro F1, but it does achieve the best macro ROC AUC. That suggests combining both models still helps the model produce better class rankings, even if it does not always improve the final top-1 prediction. This may be because the fusion method is fairly simple, the image branch is frozen, and training only runs for a single epoch.

There are also some useful recall trends behind these numbers. The multimodal model gives the best `Black` recall at `0.678`, better than text (`0.630`) and image (`0.499`), which suggests the extra image signal helps recover more true `Black` examples. It also gives the best `Green` recall at `0.926`. On the other hand, `TTR` recall drops to `0.614` in the multimodal model, compared with `0.695` for text and `0.728` for image, so the gain from combining models is not consistent across all classes.

Across all three setups, `Green` is the easiest class to classify. `Black` is the most difficult class for the image model, while `TTR` remains a weaker class for the text and multimodal models due to the ambiguity of items that can be assigned.

## Conclusion

This project shows that filename derived text is a very strong baseline for garbage classification in this dataset. The text model achieved the best overall performance, reaching `0.793` accuracy and `0.784` macro F1, while also producing the fewest test errors. That result suggests the filename-derived descriptions contain strong semantic clues about the item type, material, or disposal category, and in this setting those cues are often more informative than image features alone.

At the same time, the multimodal model still provided useful evidence that combining modalities can help. Although it did not surpass the text model on accuracy or macro F1, it achieved the highest macro ROC AUC at `0.931`, which suggests stronger overall class ranking and separation. This means the multimodal system is learning information that is useful, but that information is not yet being converted into the best final class decision consistently.

Taken together, the results suggest that the current limitation is not the idea of multimodal learning itself, but the simplicity of the present setup. The fusion method is only feature concatenation, the image encoder is frozen. A stronger fusion mechanism, more task-specific fine-tuning, and longer training would likely improve the final multimodal classifier and could allow it to outperform both single-modality baselines.
