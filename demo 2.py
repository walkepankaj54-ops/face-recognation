import cv2
import os
import csv
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

# ----------------------------
# Initialize InsightFace
# ----------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ----------------------------
# Load Known Faces
# ----------------------------
known_embeddings = []
known_names = []

photo_folder = "photos"

for file in os.listdir(photo_folder):

    if file.lower().endswith((".jpg", ".png", ".jpeg")):

        img_path = os.path.join(photo_folder, file)

        image = cv2.imread(img_path)

        faces = app.get(image)

        if len(faces) > 0:

            known_embeddings.append(faces[0].embedding)

            known_names.append(os.path.splitext(file)[0])

print("Known Faces Loaded:")
print(known_names)

# ----------------------------
# Attendance CSV
# ----------------------------

today = datetime.now().strftime("%Y-%m-%d")

csv_file = open(today + ".csv", "w", newline="")

writer = csv.writer(csv_file)

writer.writerow(["Name", "Time"])

students = known_names.copy()

# ----------------------------
# Webcam
# ----------------------------

cap = cv2.VideoCapture(0)

THRESHOLD = 0.45

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)

    for face in faces:

        embedding = face.embedding

        best_match = None
        best_score = -1

        for i, known_embedding in enumerate(known_embeddings):

            score = np.dot(embedding, known_embedding) / (
                np.linalg.norm(embedding)
                * np.linalg.norm(known_embedding)
            )

            if score > best_score:
                best_score = score
                best_match = i

        name = "Unknown"

        if best_score > THRESHOLD:
            name = known_names[best_match]

            if name in students:

                students.remove(name)

                current_time = datetime.now().strftime("%H:%M:%S")

                writer.writerow([name, current_time])

                print(f"{name} Present")

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame,
                      (x1, y1),
                      (x2, y2),
                      (0,255,0),
                      2)

        cv2.putText(frame,
                    name,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

csv_file.close()

cv2.destroyAllWindows()