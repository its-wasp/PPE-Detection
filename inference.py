import os
import cv2
from ultralytics import YOLO

def main(input_dir, output_dir, person_det_model, ppe_detection_model):
    #Load the models
    person_model = YOLO(person_det_model)  # person detection model
    ppe_model = YOLO(ppe_detection_model)  # PPE detection model

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the input directory
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        # Perform person detection on the full image
        person_results = person_model.predict(img)
        person_boxes = person_results[0].boxes.xyxy  # Get person bounding boxes

        # Iterate over each detected person
        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = map(int, box[:4])

            # Crop the person from the full image
            crop_img = img[y1:y2, x1:x2]

            # Perform PPE detection on the cropped person image
            ppe_results = ppe_model.predict(crop_img)

            # Iterate over each detected PPE item in the cropped image
            for j, ppe_box in enumerate(ppe_results[0].boxes.xyxy):
                px1, py1, px2, py2 = map(int, ppe_box[:4])
                confidence = ppe_box[-1]  # Confidence score
                class_id = int(ppe_results[0].boxes.cls[j])  # Class ID
                label = ppe_model.names[class_id]  # Get the class label

                # Map the PPE bounding box coordinates back to the full image
                full_px1, full_py1 = x1 + px1, y1 + py1
                full_px2, full_py2 = x1 + px2, y1 + py2

                # Draw the PPE bounding box and label on the original image
                cv2.rectangle(img, (full_px1, full_py1), (full_px2, full_py2), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {confidence:.2f}', (full_px1, full_py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        # Save the final annotated image
        output_img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_img_path, img)

        print(f"Processed and saved: {output_img_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on images with person and PPE detection models.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input images.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output images.")
    parser.add_argument('--person_det_model', type=str, required=True, help="Path to the trained person detection model.")
    parser.add_argument('--ppe_detection_model', type=str, required=True, help="Path to the trained PPE detection model.")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)
