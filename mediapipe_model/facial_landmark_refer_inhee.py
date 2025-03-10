import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import time
import json
import os

def calculate_au12(landmarks):
    # Lip Corner Puller - Distance between the corners of the mouth
    point_48 = np.array(landmarks[48])
    point_54 = np.array(landmarks[54])
    return dist.euclidean(point_48, point_54)

def calculate_au25_26(landmarks):
    # Lips Part and Jaw Drop - Distance between upper and lower lips and jaw
    point_51 = np.array(landmarks[51])
    point_57 = np.array(landmarks[57])
    point_8 = np.array(landmarks[8])
    lips_part = dist.euclidean(point_51, point_57)
    jaw_drop = dist.euclidean(point_57, point_8)
    return lips_part, jaw_drop

def calculate_aus(landmarks):
    au12 = calculate_au12(landmarks)
    au25, au26 = calculate_au25_26(landmarks)
    return au12, au25, au26

class FaceMeshDetector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            True,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        input_img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        input_img.flags.writeable = False
        self.results = self.face_mesh.process(input_img)
        input_img.flags.writeable = True
        img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

        self.imgH, self.imgW, self.imgC = img.shape
        self.faces = []

        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec,
                    )

                face = []
                for id, lmk in enumerate(face_landmarks.landmark):
                    x, y = int(lmk.x * self.imgW), int(lmk.y * self.imgH)
                    face.append([x, y])

                self.faces = face

        return img, self.faces

    def findReferenceValues(self, img):
        img, faces = self.findFaceMesh(img, draw=False)
        if faces:
            return calculate_aus(faces)
        return None

def main():
    detector = FaceMeshDetector()
    cap = cv2.VideoCapture(0)
    
    start_time = time.time()
    duration = 10  # 10초 동안 측정
    au_values = []
    
    print("기준 표정을 10초간 측정합니다. 카메라를 응시해주세요.")
    
    while cap.isOpened():
        success, img = cap.read()
        
        if not success:
            print("빈 프레임을 무시합니다.")
            continue
            
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time > duration:
            break
            
        img, faces = detector.findFaceMesh(img)
        if faces:
            aus = calculate_aus(faces)
            if aus:
                au_values.append(aus)
                
        cv2.putText(img, f"Measuring: {int(duration - elapsed_time)}seconds left", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Reference Face Measurement", img)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if au_values:
        mean_values = np.mean(au_values, axis=0)
        reference_data = {
            'au12': float(mean_values[0]),
            'au25': float(mean_values[1]),
            'au26': float(mean_values[2])
        }
        
        with open('reference_values.json', 'w') as f:
            json.dump(reference_data, f)
            
        print("기준 표정이 성공적으로 저장되었습니다.")
    else:
        print("얼굴이 감지되지 않았습니다. 다시 시도해주세요.")

if __name__ == "__main__":
    main()