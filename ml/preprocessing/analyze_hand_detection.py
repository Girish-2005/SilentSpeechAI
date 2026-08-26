from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd


INPUT_PATH = Path("ml/dataset/sample")

MODEL_PATH = Path(
    "ml/models/assets/hand_landmarker.task"
)

EXCLUDED_SAMPLES_PATH = Path(
    "ml/dataset/excluded_samples.txt"
)

REPORT_PATH = Path(
    "ml/dataset/hand_detection_report.csv"
)


MAX_NUM_HANDS = 2

MIN_HAND_DETECTION_CONFIDENCE = 0.3
MIN_HAND_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3


def get_sign_name(filename: str) -> str:
    """
    Convert video filename into the sign label.
    """

    name = filename.replace(".mp4", "")

    if "_(Sign_" in name:
        name = name.split("_(Sign_")[0]

    return name.replace("_", " ")


def load_excluded_samples():
    if not EXCLUDED_SAMPLES_PATH.exists():
        return set()

    excluded = set()

    with open(
        EXCLUDED_SAMPLES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            excluded.add(line)

    return excluded


def main():

    if not MODEL_PATH.exists():
        print("\nERROR: Hand Landmarker model not found.")
        print(f"Expected location: {MODEL_PATH}")
        return

    excluded_samples = load_excluded_samples()

    all_video_files = sorted(
        INPUT_PATH.glob("*.mp4")
    )

    video_files = [
        video
        for video in all_video_files
        if video.name not in excluded_samples
    ]

    print(
        f"\nTotal videos found: "
        f"{len(all_video_files)}"
    )

    print(
        f"Excluded videos: "
        f"{len(excluded_samples)}"
    )

    for sample in sorted(excluded_samples):
        print(f"  - {sample}")

    print(
        f"Videos included in analysis: "
        f"{len(video_files)}"
    )

    if not video_files:
        print(
            f"No videos found in: {INPUT_PATH}"
        )
        return

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
            model_asset_path=str(MODEL_PATH)
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

    results_data = []

    print(
        "\nStarting full video "
        "hand detection analysis...\n"
    )
    
    global_timestamp_ms = 0

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        for video_path in video_files:

            print(
                f"Processing: "
                f"{video_path.name}"
            )

            cap = cv2.VideoCapture(
                str(video_path)
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            frame_index = 0
            detected_frames = 0
            total_hands = 0

            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    global_timestamp_ms,
)

                frame_duration_ms = int(1000 / fps)

                if frame_duration_ms <= 0:
                    frame_duration_ms = 1

                global_timestamp_ms += frame_duration_ms

                hand_count = len(
                    result.hand_landmarks
                )

                if hand_count > 0:
                    detected_frames += 1
                    total_hands += hand_count

                frame_index += 1

            cap.release()

            detection_rate = 0

            if frame_index > 0:

                detection_rate = (
                    detected_frames
                    / frame_index
                ) * 100

            sign_name = get_sign_name(
                video_path.name
            )

            results_data.append(
                {
                    "video": video_path.name,
                    "sign": sign_name,
                    "total_frames": frame_index,
                    "frames_with_hands": (
                        detected_frames
                    ),
                    "total_hands_detected": (
                        total_hands
                    ),
                    "detection_rate": round(
                        detection_rate,
                        2,
                    ),
                }
            )

            print(
                f"  Frames: {frame_index}"
            )

            print(
                f"  Frames with hands: "
                f"{detected_frames}"
            )

            print(
                f"  Detection rate: "
                f"{detection_rate:.2f}%\n"
            )

    df = pd.DataFrame(
        results_data
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        REPORT_PATH,
        index=False,
    )

    print("=" * 60)

    print(
        "HAND DETECTION ANALYSIS COMPLETE"
    )

    print("=" * 60)

    print(
        f"\nOverall detection rate: "
        f"{df['frames_with_hands'].sum() / df['total_frames'].sum() * 100:.2f}%"
    )

    print(
        "\nDetection rate by sign:"
    )

    print(
        df.groupby("sign")[
            "detection_rate"
        ].mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        f"\nReport saved to: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()