from pathlib import Path

import cv2
import mediapipe as mp


INPUT_PATH = Path("ml/dataset/inspection_frames")
OUTPUT_PATH = Path("ml/dataset/hand_detection_preview")

MODEL_PATH = Path(
    "ml/models/assets/hand_landmarker.task"
)

MAX_NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


def draw_landmarks(image, landmarks):
    """Draw detected hand landmarks and connections."""

    height, width, _ = image.shape

    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    for connection in connections:
        start = connection.start
        end = connection.end

        start_landmark = landmarks[start]
        end_landmark = landmarks[end]

        start_point = (
            int(start_landmark.x * width),
            int(start_landmark.y * height),
        )

        end_point = (
            int(end_landmark.x * width),
            int(end_landmark.y * height),
        )

        cv2.line(
            image,
            start_point,
            end_point,
            (0, 255, 0),
            2,
        )

    for landmark in landmarks:
        point = (
            int(landmark.x * width),
            int(landmark.y * height),
        )

        cv2.circle(
            image,
            point,
            4,
            (0, 0, 255),
            -1,
        )


def main():
    if not MODEL_PATH.exists():
        print("\nERROR: Hand Landmarker model not found.")
        print(f"Expected location: {MODEL_PATH}")
        return

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=(
            MIN_HAND_DETECTION_CONFIDENCE
        ),
        min_hand_presence_confidence=(
            MIN_HAND_PRESENCE_CONFIDENCE
        ),
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    image_files = sorted(INPUT_PATH.rglob("*.jpg"))

    if not image_files:
        print(f"No images found in: {INPUT_PATH}")
        return

    total_images = 0
    detected_images = 0
    total_hands = 0

    print("\nStarting MediaPipe HandLandmarker validation...\n")

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        for image_path in image_files:

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Could not read: {image_path}")
                continue

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb,
            )

            result = landmarker.detect(mp_image)

            total_images += 1

            hand_count = len(result.hand_landmarks)

            if hand_count > 0:
                detected_images += 1
                total_hands += hand_count

                for hand_landmarks in result.hand_landmarks:
                    draw_landmarks(
                        image,
                        hand_landmarks,
                    )

            relative_path = image_path.relative_to(
                INPUT_PATH
            )

            output_file = OUTPUT_PATH / relative_path

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            cv2.imwrite(
                str(output_file),
                image,
            )

            print(
                f"{relative_path} -> "
                f"{hand_count} hand(s) detected"
            )

    print("\n" + "=" * 60)
    print("HAND DETECTION VALIDATION COMPLETE")
    print("=" * 60)

    print(f"Total images tested: {total_images}")
    print(
        f"Images with hands detected: "
        f"{detected_images}"
    )
    print(f"Total hands detected: {total_hands}")

    if total_images > 0:
        detection_rate = (
            detected_images / total_images
        ) * 100

        print(
            f"Detection rate: "
            f"{detection_rate:.2f}%"
        )

    print(
        f"\nPreview saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()