import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os
import time
import json

base_options = mp_python.BaseOptions(model_asset_path='./gesture_recognizer.task')
options = mp_vision.GestureRecognizerOptions(base_options=base_options)
recognizer = mp_vision.GestureRecognizer.create_from_options(options)

def calculate_au6(landmarks):
    # Cheek Raiser - Distance between outer corner of the eyes and upper cheek
    point_263 = np.array(landmarks[263])  # Outer corner of the right eye
    point_374 = np.array(landmarks[374])  # Upper cheek below the right eye
    return dist.euclidean(point_263, point_374)

def calculate_au12(landmarks):
    # Lip Corner Puller - Distance between the corners of the mouth
    point_48 = np.array(landmarks[48])
    point_54 = np.array(landmarks[54])
    return dist.euclidean(point_48, point_54)

def calculate_au16(landmarks):
    # Lower Lip Depressor - Distance between the lower lip and chin
    point_57 = np.array(landmarks[57])
    point_8 = np.array(landmarks[8])
    return dist.euclidean(point_57, point_8)

def calculate_au25_26(landmarks):
    # Lips Part and Jaw Drop - Distance between upper and lower lips and jaw
    point_51 = np.array(landmarks[51])
    point_57 = np.array(landmarks[57])
    point_8 = np.array(landmarks[8])
    lips_part = dist.euclidean(point_51, point_57)
    jaw_drop = dist.euclidean(point_57, point_8)
    return lips_part, jaw_drop

def calculate_aus(landmarks):
    au6 = calculate_au6(landmarks)
    au12 = calculate_au12(landmarks)
    au16 = calculate_au16(landmarks)
    au25, au26 = calculate_au25_26(landmarks)
    return au6, au12, au16, au25, au26


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

        # Facemesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            True,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )
        # hand gesture recognizer
        self.recognizer = recognizer
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.stored_scores = []
        
        try:
            with open('reference_values.json', 'r') as f:
                self.reference_values = json.load(f)
        except FileNotFoundError:
            print("경고: reference_values.json 파일을 찾을 수 없습니다.")
            print("facial_landmark_refer.py를 먼저 실행하여 기준 표정을 설정해주세요.")
            exit()

    def calculate_happiness_score(self, current_aus):
        au6, au12, au16, au25, au26 = current_aus
        ref_au6 = self.reference_values['au6']
        ref_au12 = self.reference_values['au12']
        ref_au16 = self.reference_values['au16']
        ref_au25 = self.reference_values['au25']
        ref_au26 = self.reference_values['au26']
        
        # 각 AU의 변화율 계산
        au6_change = (au6 - ref_au6) / ref_au6 * 100
        au12_change = (au12 - ref_au12) / ref_au12 * 100
        au16_change = (au16 - ref_au16) / ref_au16 * 100
        au25_change = (au25 - ref_au25) / ref_au25 * 100
        au26_change = (au26 - ref_au26) / ref_au26 * 100
        
        # 가중치 적용 (총합 100%)
        score = 80 + (
            0.35 * au6_change +   # 35%
            0.35 * au12_change +  # 35%
            0.10 * au16_change +  # 10%
            0.10 * au25_change +  # 10%
            0.10 * au26_change    # 10%
        )
        
        return np.clip(score, 0, 100)  # 0-100 범위로 제한

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
                
        gesture_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_img)
        recognition_result = self.recognizer.recognize(gesture_image)

        return img, self.faces, recognition_result.gestures

    def findSmileScore(self, img):
        img, faces, _ = self.findFaceMesh(img, draw=False)
        if faces:
            aus = calculate_aus(faces)
            if aus:
                score = self.calculate_happiness_score(aus)
                self.stored_scores.append(score)
            return score
        return None


# sample run of the module
def main():
    detector = FaceMeshDetector()

    cap = cv2.VideoCapture(0)

    start_time = time.time()
    duration = 10  # 10초 측정
    
    print("행복도를 10초간 측정합니다. 카메라를 응시해주세요.")

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time > duration:
            break

        img, faces, _ = detector.findFaceMesh(img)  # Draw mesh
        if faces:
            score = detector.findSmileScore(img)
            if score is not None:
                cv2.putText(img, f"Current score: {score:.2f}", 
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(img, f"Measuring: {int(duration - elapsed_time)}seconds left", 
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Happiness Score Measurement", img)

        # Press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if detector.stored_scores:
        # 가중 평균 계산 (최근 값에 더 높은 가중치 부여)
        weights = np.linspace(1, 2, len(detector.stored_scores))
        weights = weights / np.sum(weights)  # 정규화
        final_score = np.average(detector.stored_scores, weights=weights)
        print(f"\n최종 행복도 점수: {final_score:.2f}/100")
    else:
        print("측정에 실패했습니다. 다시 시도해주세요.")

if __name__ == "__main__":
    main()