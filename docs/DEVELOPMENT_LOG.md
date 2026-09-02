# SilentSpeech AI — Development Log

## Milestone 2.1 — Hand Landmark Extraction

### Status
Completed

### Objective
Extract hand landmarks from the ISL video dataset using MediaPipe HandLandmarker.

### Dataset Validation
The initial ISL dataset contained 12 videos across 6 sign classes:

- Hello
- Help
- No
- Please
- Thank You
- Yes

One sample, `Yes_(Sign_2).mp4`, was manually identified as unsuitable because the subject does not perform the intended hand sign. The subject primarily moves without producing the target sign.

Therefore:

- Total videos: 12
- Excluded videos: 1
- Valid videos: 11

### Hand Detection Results

After excluding `Yes_(Sign_2).mp4`:

- Overall hand detection rate: 50.99%
- Highest detection rate: Please — 57.76%
- Lowest detection rate: Help — 40.48%

The detection pipeline was validated using MediaPipe Tasks API / HandLandmarker.

### Landmark Representation

For every detected hand:

- 21 hand landmarks are extracted.
- Each landmark contains X, Y and Z coordinates.
- Total numerical landmark features per hand: 63.
- Maximum supported hands per frame: 2.

The generated CSV representation contains:

- frame_index
- timestamp_ms
- hand_index
- handedness
- handedness_confidence
- 21 × (X, Y, Z) landmark values

Total CSV columns: 68.

### Validation Sample

`Help.csv` was successfully generated and validated:

- Video frames: 84
- Frames containing hands: 34
- Hand records: 53
- Right-hand records: 33
- Left-hand records: 20
- Missing values: 0
- CSV columns: 68

### Important Observation

Some frames containing two detected hands were classified by MediaPipe with the same handedness label, for example:

- Frame 27 → [Right, Right]
- Frame 33 → [Right, Right]

Therefore, the sequence-building stage must not blindly rely on the handedness label to assign physical left/right feature slots.

The raw handedness information will be preserved for analysis, while sequence construction will initially use detected hand index/slot information.

---

## Milestone 2.2 — Sequence Construction

### Status
In Progress

### Objective

Convert per-hand landmark CSV data into frame-aligned temporal sequences suitable for the downstream ML pipeline.

### Design

The sequence builder must reconstruct every original video frame, including frames where no hand was detected.

For each frame:

- Up to 2 hands are represented.
- Each hand contains 21 × 3 = 63 landmark values.
- Maximum landmark representation per frame = 126 values.
- Missing hands are represented using a detection mask.

Expected representation:

- Landmark tensor: `(number_of_frames, 126)`
- Detection mask: `(number_of_frames, 2)`

### Normalization

Landmarks will initially be normalized relative to the wrist landmark (landmark 0):

`normalized_landmark = landmark - wrist`

This reduces dependency on the absolute position of the hand within the camera frame.

### Design Principle

The sequence builder must preserve temporal ordering and missing-frame information rather than compressing the CSV rows directly into a sequence.

This representation will later be evaluated before fixed-length temporal resampling and model training are introduced.