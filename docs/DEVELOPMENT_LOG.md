# Development Log

## Milestone 1 — Dataset Validation

### Status
Completed

### Work Completed

- Initialized the SilentSpeech AI repository.
- Established the project directory structure.
- Verified the ISL sample dataset.
- Analyzed video count, FPS, resolution, duration, and frame counts.
- Extracted representative frames for visual inspection.
- Migrated hand detection implementation to the MediaPipe Tasks API.
- Validated hand detection across the complete video samples.
- Identified an invalid dataset sample.
- Added `Yes_(Sign_2).mp4` to the excluded sample list.
- Re-ran hand detection analysis using only valid samples.

### Dataset Findings

Original dataset:
- 12 videos
- 6 classes

After validation:
- 11 valid videos
- 6 classes
- 1 excluded sample

Excluded sample:

`Yes_(Sign_2).mp4`

Reason:

The video does not contain a meaningful hand-sign gesture despite being labeled as `Yes`.

### Hand Detection Baseline

Overall frame-level detection rate on valid samples:

**50.99%**

Best performing class:

**Please — 57.76%**

Lowest performing class:

**Help — 40.48%**

### Technical Decision

The project will use MediaPipe's Tasks API and support detection of up to two hands.

Individual frames without detected hands will not automatically cause the entire video to be discarded.

The landmark extraction pipeline will handle missing observations before sequence generation.

### Important Limitation

The current dataset is too small for reliable production model training. It is currently being used to validate the processing and inference pipeline.

### Next Milestone

Milestone 2:

**Hand Landmark Extraction and Sequence Generation**