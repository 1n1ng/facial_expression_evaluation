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

def calculate_face_size(landmarks):
    """얼굴 전체 크기를 측정하는 함수 (정규화에 사용)"""
    try:
        # 얼굴 가로 크기 (왼쪽 귀에서 오른쪽 귀까지)
        left_ear = np.array(landmarks[234])
        right_ear = np.array(landmarks[454])
        face_width = dist.euclidean(left_ear, right_ear)
        
        # 얼굴 세로 크기 (턱에서 이마까지)
        chin = np.array(landmarks[152])
        forehead = np.array(landmarks[10])
        face_height = dist.euclidean(chin, forehead)
        
        # 얼굴 전체 크기 (가로와 세로의 평균)
        return (face_width + face_height) / 2
    except IndexError:
        print("랜드마크 인덱스 오류: 얼굴 크기 계산 실패")
        return 1.0  # 오류 시 기본값 반환

def calculate_au12(landmarks, face_size):
    # Lip Corner Puller - 정규화된 거리
    try:
        point_48 = np.array(landmarks[48])
        point_54 = np.array(landmarks[54])
        absolute_distance = dist.euclidean(point_48, point_54)
        # 얼굴 크기로 정규화
        return absolute_distance / face_size
    except IndexError:
        print("랜드마크 인덱스 오류: 충분한 얼굴 특징점이 감지되지 않았습니다.")
        return 0.0

def calculate_au25_26(landmarks, face_size):
    # Lips Part and Jaw Drop - 정규화된 거리
    try:
        point_51 = np.array(landmarks[51])
        point_57 = np.array(landmarks[57])
        point_8 = np.array(landmarks[8])
        
        lips_part = dist.euclidean(point_51, point_57) / face_size
        jaw_drop = dist.euclidean(point_57, point_8) / face_size
        
        return lips_part, jaw_drop
    except IndexError:
        print("랜드마크 인덱스 오류: 충분한 얼굴 특징점이 감지되지 않았습니다.")
        return 0.0, 0.0

def calculate_aus(landmarks):
    face_size = calculate_face_size(landmarks)
    au12 = calculate_au12(landmarks, face_size)
    au25, au26 = calculate_au25_26(landmarks, face_size)
    return au12, au25, au26

class FaceMeshDetector:
    
    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.3,  # 신뢰도 임계값 낮춤
        min_tracking_confidence=0.3,   # 트래킹 신뢰도 임계값 낮춤
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

    def calculate_smile_score(self, current_aus):
        au12, au25, au26 = current_aus
        ref_au12 = self.reference_values['au12']
        ref_au25 = self.reference_values['au25']
        ref_au26 = self.reference_values['au26']
        
        # 각 AU의 변화율 계산
        au12_change = (au12 - ref_au12) / ref_au12 * 100
        au25_change = (au25 - ref_au25) / ref_au25 * 100
        au26_change = (au26 - ref_au26) / ref_au26 * 100
        
        # 변화율이 양수일 때는 증폭시키고, 음수일 때는 약화시킴
        if au12_change > 0:
            au12_change = au12_change * 2.5  # 미소 확대 효과 증폭
        
        if au25_change > 0:
            au25_change = au25_change * 1.5
            
        if au26_change > 0:
            au26_change = au26_change * 1.5
            
        # 기본 점수를 50이 아닌 60으로 시작 (역치 낮추기)
        score = 60 + (
            0.60 * au12_change +  # 60%
            0.20 * au25_change +  # 20%
            0.20 * au26_change    # 20%
        )
        
        # 매우 작은 미소도 감지
        if au12_change > 0:
            score += 10  # 미소의 방향이 맞다면 기본 점수 추가
            
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
        if faces and len(faces) >= 468:  # 충분한 랜드마크가 있는지 확인
            try:
                aus = calculate_aus(faces)
                if aus:
                    score = self.calculate_smile_score(aus)
                    self.stored_scores.append(score)
                    return score
            except Exception as e:
                print(f"얼굴 특징점 계산 중 오류 발생: {e}")
                return None
        return None


# sample run of the module
def main():
    detector = FaceMeshDetector()

    cap = cv2.VideoCapture(0)
    
    # 카메라 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 웜업 - 카메라 초기화 시간 부여
    print("카메라 초기화 중...")
    for _ in range(10):
        ret, _ = cap.read()
        if not ret:
            print("카메라 초기화 실패, 다시 시도하세요.")
            return
        time.sleep(0.1)

    start_time = time.time()
    duration = 10  # 10초 측정
    
    print("미소 정도를 10초간 측정합니다. 카메라를 응시해주세요.")

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
        
        # 얼굴 감지 상태 표시
        if faces:
            cv2.putText(img, "Face Detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            score = detector.findSmileScore(img)
            if score is not None:
                # 색상을 점수에 따라 변경 (낮은 점수도 녹색으로 표시)
                color = (0, 255, 0)  # 항상 녹색으로 표시
                
                # 점수 표현 방식 수정 - 점수가 낮아도 긍정적인 메시지 표시
                if score >= 80:
                    message = f"Excellent! Score: {score:.1f}"
                elif score >= 60:
                    message = f"Great! Score: {score:.1f}"
                elif score >= 40:
                    message = f"Good! Score: {score:.1f}"
                else:
                    message = f"Start! Score: {score:.1f}"
                    
                cv2.putText(img, message, 
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(img, "No Face Detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(img, f"Measuring: {int(duration - elapsed_time)}seconds left", 
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Smile Score Measurement", img)

        # Press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if detector.stored_scores:
        # 가중 평균 계산 (최근 값에 더 높은 가중치 부여)
        weights = np.linspace(1, 2, len(detector.stored_scores))
        weights = weights / np.sum(weights)  # 정규화
        
        # 최고 점수와 평균 점수 계산
        max_score = np.max(detector.stored_scores)
        final_score = np.average(detector.stored_scores, weights=weights)
        
        # 최고 점수에 가중치를 두어 최종 점수 계산 (최고 점수 50%, 가중 평균 50%)
        biased_final_score = 0.5 * max_score + 0.5 * final_score
        
        print(f"\n최종 미소 점수: {biased_final_score:.2f}/100")
        
        # 긍정적인 피드백 메시지
        if biased_final_score >= 80:
            print("훌륭합니다! 아주 좋은 미소를 지으셨어요!")
        elif biased_final_score >= 60:
            print("잘 하셨어요! 좋은 미소였습니다!")
        elif biased_final_score >= 40:
            print("좋아요! 미소가 보였습니다!")
        else:
            print("잘 하셨어요! 다음에는 더 환하게 웃어볼까요?")
    else:
        print("측정에 실패했습니다. 다시 시도해주세요.")

if __name__ == "__main__":
    main()