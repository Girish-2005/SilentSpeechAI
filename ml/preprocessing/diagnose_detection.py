import os
import cv2
import mediapipe as mp


VIDEO_PATH = "ml/dataset/sample/Yes_(Sign_2).mp4"


CONFIGURATIONS = [
    {
        "name": "default",
        "num_hands": 2,
        "min_hand_detection_confidence": 0.5,
        "min_hand_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },
    {
        "name": "medium",
        "num_hands": 2,
        "min_hand_detection_confidence": 0.3,
        "min_hand_presence_confidence": 0.3,
        "min_tracking_confidence": 0.3,
    },
    {
        "name": "sensitive",
        "num_hands": 2,
        "min_hand_detection_confidence": 0.1,
        "min_hand_presence_confidence": 0.1,
        "min_tracking_confidence": 0.1,
    },
]


MODEL_PATH = "ml/models/assets/hand_landmarker.task"


def analyze_configuration(config):
    print("\n" + "=" * 60)
    print(f"Testing configuration: {config['name']}")
    print("=" * 60)

    base_options = mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=config["num_hands"],
        min_hand_detection_confidence=config[
            "min_hand_detection_confidence"
        ],
        min_hand_presence_confidence=config[
            "min_hand_presence_confidence"
        ],
        min_tracking_confidence=config[
            "min_tracking_confidence"
        ],
    )

    cap = cv2.VideoCapture(VIDEO_PATH)

    total_frames = 0
    detected_frames = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                break

            total_frames += 1

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Frame index ensures monotonically increasing timestamps
            timestamp_ms = total_frames * 40

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.hand_landmarks:
                detected_frames += 1

    cap.release()

    detection_rate = (
        detected_frames / total_frames * 100
        if total_frames > 0
        else 0
    )

    print(f"Total frames: {total_frames}")
    print(f"Frames with hands: {detected_frames}")
    print(f"Detection rate: {detection_rate:.2f}%")

    return {
        "name": config["name"],
        "rate": detection_rate
    }


def main():

    print("\nStarting hand detection configuration diagnosis...")
    print(f"Video: {VIDEO_PATH}")

    if not os.path.exists(VIDEO_PATH):
        print(f"\nERROR: Video not found: {VIDEO_PATH}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found: {MODEL_PATH}")
        return

    results = []

    for config in CONFIGURATIONS:
        result = analyze_configuration(config)
        results.append(result)

    print("\n" + "=" * 60)
    print("CONFIGURATION COMPARISON")
    print("=" * 60)

    for result in results:
        print(
            f"{result['name']:12} "
            f"-> {result['rate']:.2f}%"
        )

    best_result = max(
        results,
        key=lambda x: x["rate"]
    )

    print("\nBest configuration:")
    print(
        f"{best_result['name']} "
        f"with {best_result['rate']:.2f}% detection rate"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()