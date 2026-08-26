# Dataset Documentation

## 1. Dataset Overview

The SilentSpeech AI project uses an Indian Sign Language (ISL) video dataset for development and model validation.

The current development sample contains short videos representing six sign classes.

## 2. Dataset Statistics

| Property | Value |
|---|---:|
| Original videos | 12 |
| Excluded videos | 1 |
| Valid videos | 11 |
| Unique sign classes | 6 |
| FPS | 25 |
| Resolution | 1920 × 1080 |
| Average duration | 3.27 seconds |
| Average frame count | 81.75 |

## 3. Sign Distribution

| Sign | Videos |
|---|---:|
| Please | 4 |
| No | 2 |
| Thank You | 2 |
| Yes | 1 valid |
| Help | 1 |
| Hello | 1 |

## 4. Excluded Sample

### Yes_(Sign_2).mp4

This sample is excluded from model training and evaluation.

Although the filename indicates the `Yes` class, visual inspection showed that the subject does not perform a meaningful hand-sign gesture in this video. The subject primarily moves without producing the expected hand sign.

The original sample is retained for dataset traceability but is excluded from the valid training/evaluation set.

## 5. Video Characteristics

All analyzed videos have:

- 25 FPS
- 1920 × 1080 resolution
- approximately 2.68–3.96 seconds duration
- approximately 67–99 frames

## 6. Hand Detection Validation

MediaPipe Hand Landmarker using the Tasks API was evaluated on every frame of the 11 valid videos.

### Overall Result

- Total valid frames: 898
- Frames with detected hands: 458
- Overall frame-level detection rate: 50.99%

### Detection Rate by Sign

| Sign | Detection Rate |
|---|---:|
| Please | 57.76% |
| Yes | 53.49% |
| No | 48.27% |
| Hello | 47.30% |
| Thank You | 43.80% |
| Help | 40.48% |

## 7. Interpretation

The current sample demonstrates that hand landmarks can be extracted from the dataset, but frame-level detection is inconsistent.

This indicates that the preprocessing pipeline must account for frames where hand landmarks are unavailable.

The project will therefore not directly discard entire videos when individual frames lack detections. Instead, landmark extraction and temporal sequence construction will be designed to handle missing observations consistently.

## 8. Dataset Limitations

The current sample is a development/validation dataset and is too small for a production-quality sign recognition model.

There are currently only 11 valid videos across six classes, with some classes represented by a single video.

A larger and more balanced ISL dataset will be required for meaningful CNN/LSTM training and generalization evaluation.

## 9. Current Status

Milestone 1 dataset validation is complete.

Next milestone:

**Landmark extraction and normalized sequence generation.**