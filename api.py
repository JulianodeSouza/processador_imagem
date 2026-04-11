from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def get_face_metrics(image):
    h, w, _ = image.shape
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    faces_data = []

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            faces_data.append({
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "center_x": x + width // 2,
                "center_y": y + height // 2
            })

    return faces_data


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces = get_face_metrics(image)

    return {
        "faces_detected": len(faces),
        "faces": faces
    }