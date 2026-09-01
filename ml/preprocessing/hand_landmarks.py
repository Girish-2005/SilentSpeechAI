from pathlib import Path
import csv

import cv2
import mediapipe as mp


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATASET_PATH = Path("ml/dataset/sample")
MODEL_PATH = Path("ml/models/assets/hand_landmarker.task")
OUTPUT_PATH = Path("ml/dataset/landmarks")


# ---------------------------------------------------------
# MediaPipe configuration
# ---------------------------------------------------------

MAX_NUM_HANDS = 2

MIN_HAND_DETECTION_CONFIDENCE = 0.3
MIN_HAND_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3


# ---------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------

def load_excluded_samples():
    """Load videos that must not be processed."""

    exclusion_file = (
        Path("ml/dataset/excluded_samples.txt")
    )

    if not exclusion_file.exists():
        return set()

    excluded = set()

    with open(
        exclusion_file,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            excluded.add(line)

    return excluded


def get_sign_name(filename):
    """
    Convert a video filename into its sign label.

    Examples:
        Hello.mp4 -> Hello
        Thank_You.mp4 -> Thank You
        Please_(Sign_2).mp4 -> Please
    """

    name = Path(filename).stem

    if "_(Sign_" in name:
        name = name.split("_(Sign_")[0]

    return name.replace("_", " ")


# ---------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------

def extract_video_landmarks(
    video_path,
    landmarker,
):
    """
    Extract hand landmarks from every frame of a video.

    Returns a list containing one dictionary per frame.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    frame_records = []

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        timestamp_ms = int(
            (frame_index / fps) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        frame_record = {
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "hands": [],
        }

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):

            handedness = "unknown"
            handedness_score = 0.0

            if (
                result.handedness
                and hand_index < len(result.handedness)
                and result.handedness[hand_index]
            ):

                category = (
                    result.handedness[
                        hand_index
                    ][0]
                )

                handedness = category.category_name
                handedness_score = (
                    category.score
                )

            landmark_values = []

            for landmark in landmarks:

                landmark_values.extend(
                    [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                    ]
                )

            frame_record["hands"].append(
                {
                    "handedness": handedness,
                    "confidence": handedness_score,
                    "landmarks": landmark_values,
                }
            )

        frame_records.append(
            frame_record
        )

        frame_index += 1

    cap.release()

    return frame_records


# ---------------------------------------------------------
# CSV output
# ---------------------------------------------------------

def save_landmarks_csv(
    records,
    output_file,
):
    """
    Save extracted landmarks to CSV.

    One row represents one detected hand.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = [
        "frame_index",
        "timestamp_ms",
        "hand_index",
        "handedness",
        "handedness_confidence",
    ]

    for landmark_index in range(21):

        header.extend(
            [
                f"landmark_{landmark_index}_x",
                f"landmark_{landmark_index}_y",
                f"landmark_{landmark_index}_z",
            ]
        )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)

        for record in records:

            for hand_index, hand in enumerate(
                record["hands"]
            ):

                row = [
                    record["frame_index"],
                    record["timestamp_ms"],
                    hand_index,
                    hand["handedness"],
                    hand["confidence"],
                ]

                row.extend(
                    hand["landmarks"]
                )

                writer.writerow(row)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def process_video(
    video_path,
    landmarker,
):

    sign_name = get_sign_name(
        video_path.name
    )

    print(
        f"\nProcessing: "
        f"{video_path.name}"
    )

    records = extract_video_landmarks(
        video_path,
        landmarker,
    )

    total_frames = len(records)

    frames_with_hands = sum(
        1
        for record in records
        if record["hands"]
    )

    total_hands = sum(
        len(record["hands"])
        for record in records
    )

    detection_rate = (
        frames_with_hands / total_frames * 100
        if total_frames
        else 0.0
    )

    output_file = (
        OUTPUT_PATH
        / f"{video_path.stem}.csv"
    )

    save_landmarks_csv(
        records,
        output_file,
    )

    print(
        f"Sign: {sign_name}"
    )

    print(
        f"Frames: {total_frames}"
    )

    print(
        f"Frames with hands: "
        f"{frames_with_hands}"
    )

    print(
        f"Total hands detected: "
        f"{total_hands}"
    )

    print(
        f"Detection rate: "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Saved: {output_file}"
    )


def main():

    print(
        "=" * 60
    )

    print(
        "SILENT SPEECH AI"
    )

    print(
        "HAND LANDMARK EXTRACTION"
    )

    print(
        "=" * 60
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nMediaPipe model not found:\n"
            f"{MODEL_PATH}"
        )

    excluded_samples = (
        load_excluded_samples()
    )

    video_files = [
        DATASET_PATH / "Help.mp4"
    ]

    video_files = [
        video
        for video in video_files
        if video.name not in excluded_samples
    ]

    print(
        f"\nVideos to process: "
        f"{len(video_files)}"
    )

    print(
        "Excluded samples: "
        f"{len(excluded_samples)}"
    )

    # -----------------------------------------------------
    # MediaPipe Tasks API
    # -----------------------------------------------------

    BaseOptions = mp.tasks.BaseOptions

    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(
                MODEL_PATH
            )
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=(
            MIN_HAND_DETECTION_CONFIDENCE
        ),
        min_hand_presence_confidence=(
            MIN_HAND_PRESENCE_CONFIDENCE
        ),
        min_tracking_confidence=(
            MIN_TRACKING_CONFIDENCE
        ),
    )

    # -----------------------------------------------------
    # Process videos
    # -----------------------------------------------------

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        for video_path in video_files:

            process_video(
                video_path,
                landmarker,
            )

    print(
        "\n" + "=" * 60
    )

    print(
        "LANDMARK EXTRACTION COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()