import os
import datetime
import json
import cv2
import numpy as np
import time
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags
from io import BytesIO
from tflite_runtime.interpreter import Interpreter
import mediapipe as mp
import hdbscan
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
import face_recognition

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# 글로벌 변수: 마지막 업로드(추론 완료) 시간
LAST_UPLOAD_TIME = None

# 클러스터 라벨 매핑 파일 경로
CLUSTER_LABELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_labels.json")

#############################################
# 기본 경로 및 폴더 설정
#############################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    logging.info("DATA_DIR 생성: %s", DATA_DIR)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#############################################
# 클러스터 라벨 매핑 로드/저장 함수
#############################################
def load_cluster_labels():
    if os.path.exists(CLUSTER_LABELS_FILE):
        with open(CLUSTER_LABELS_FILE, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            logging.info("클러스터 라벨 매핑 로드됨: %s", mapping)
            return mapping
    else:
        return {}

def save_cluster_labels(mapping):
    with open(CLUSTER_LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)

#############################################
# 라벨 로드 함수 (labels_skin_cancer.txt)
#############################################
def load_labels_txt(txt_file_path):
    with open(txt_file_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels

LABELS_FILE = os.path.join(BASE_DIR, "model", "skin", "labels_skin_cancer.txt")
labels_skin = load_labels_txt(LABELS_FILE)
logging.info("Loaded Skin Cancer Labels: %s", labels_skin)

# 영문 라벨 → 한글 매핑
KOREAN_LABEL_MAP = {
    "akiec": "광선각화증 및 상피내암",
    "bcc": "기저세포암",
    "bkl": "양성 각화증 유사 병변",
    "df": "피부 섬유종",
    "mel": "흑색종",
    "nv": "멜라닌 모반",
    "vasc": "혈관 병변"
}

#############################################
# TFLite 피부 질환 분류 모델 로드
#############################################
SKIN_MODEL_PATH = os.path.join(BASE_DIR, "model", "skin", "skin_cancer_best_model.tflite")
interpreter_skin = Interpreter(model_path=SKIN_MODEL_PATH)
interpreter_skin.allocate_tensors()
skin_input_details = interpreter_skin.get_input_details()
skin_output_details = interpreter_skin.get_output_details()
logging.info("TFLite 모델 로드 완료: %s", SKIN_MODEL_PATH)

def run_skin_inference(image_path):
    img = cv2.imread(image_path)
    input_shape = skin_input_details[0]['shape']
    img_resized = cv2.resize(img, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    interpreter_skin.set_tensor(skin_input_details[0]['index'], input_data)
    interpreter_skin.invoke()
    output_data = np.copy(interpreter_skin.get_tensor(skin_output_details[0]['index']))
    pred_index = int(np.argmax(output_data[0]))
    confidence = float(output_data[0][pred_index])
    pred_label = labels_skin[pred_index] if pred_index < len(labels_skin) else "Unknown"
    pred_label_kor = KOREAN_LABEL_MAP.get(pred_label, pred_label)
    return {"predicted_label": pred_label,
            "predicted_label_kor": pred_label_kor,
            "confidence": confidence,
            "raw_output": output_data.tolist()}

#############################################
# face_recognition 기반 얼굴 인식 및 임베딩 함수
#############################################
def run_face_recognition(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_encodings = [enc.tolist() for enc in face_encodings]
        logging.info("파일 %s: %d 개의 얼굴 인식", image_path, len(face_encodings))
        return {"face_count": len(face_encodings), "face_embeddings": face_encodings}
    except Exception as e:
        logging.error("face_recognition 오류 (%s): %s", image_path, e)
        return {"face_count": 0, "face_embeddings": []}

#############################################
# MediaPipe Hands 손 검출 함수
#############################################
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

def run_hand_detection(image_path):
    img = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    hand_landmarks_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": getattr(lm, 'visibility', 1.0)}
                         for lm in hand_landmarks.landmark]
            hand_landmarks_list.append(landmarks)
    logging.info("파일 %s: %d 개의 손 검출", image_path, len(hand_landmarks_list))
    return {"hand_count": len(hand_landmarks_list), "hand_landmarks": hand_landmarks_list}

#############################################
# MediaPipe Pose 신체 검출 함수
#############################################
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True)

def run_body_detection(image_path):
    img = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)
    landmarks = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
    logging.info("파일 %s: %d 개의 신체 랜드마크 검출", image_path, len(landmarks))
    return {"landmark_count": len(landmarks), "landmarks": landmarks}

#############################################
# L2 정규화 함수
#############################################
def normalize_embedding(embedding):
    arr = np.array(embedding)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm

#############################################
# 모든 추론 실행 함수
#############################################
def run_all_inferences(image_path):
    results = {}
    times = {}
    
    t0 = time.perf_counter()
    results["face"] = run_face_recognition(image_path)
    times["face"] = time.perf_counter() - t0
    
    t1 = time.perf_counter()
    results["hand"] = run_hand_detection(image_path)
    times["hand"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    results["body"] = run_body_detection(image_path)
    times["body"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    results["skin"] = run_skin_inference(image_path)
    times["skin"] = time.perf_counter() - t3

    total_time = sum(times.values())
    results["inference_times"] = times
    results["total_inference_time"] = total_time
    logging.info("파일 %s: 전체 추론 완료 (총 소요 시간: %.3f초)", image_path, total_time)
    return results

#############################################
# 도우미 함수: 파일명으로 촬영 시간 추출
#############################################
def get_timestamp_from_filename(folder, filename):
    try:
        folder_date = datetime.datetime.strptime(folder, "%Y-%m-%d").date()
    except Exception:
        folder_date = datetime.datetime.now().date()
    current_time = datetime.datetime.now().time()
    return datetime.datetime.combine(folder_date, current_time)

#############################################
# 얼굴 클러스터링 함수: HDBSCAN 및 클러스터 라벨 적용
#############################################
def cluster_face_images():
    items = []
    logging.debug("클러스터링 함수 시작")
    # DATA_DIR 내 모든 폴더 및 파일 순회
    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if allowed_file(file):
                    base_filename = file.rsplit('.', 1)[0]
                    json_filepath = os.path.join(folder_path, base_filename + ".json")
                    if os.path.exists(json_filepath):
                        with open(json_filepath, "r", encoding="utf-8") as jf:
                            inference_info = json.load(jf)
                        if inference_info and inference_info.get("face", {}).get("face_count", 0) > 0:
                            ts = get_timestamp_from_filename(folder, file)
                            # 이미지 내 모든 얼굴 임베딩 처리
                            for embedding in inference_info["face"]["face_embeddings"]:
                                norm_embedding = normalize_embedding(embedding)
                                items.append({
                                    "folder": folder,
                                    "filename": file,
                                    "filepath": f"/data/{folder}/{file}",
                                    "inference": inference_info,
                                    "timestamp": ts,
                                    "embedding": norm_embedding
                                })
    if not items:
        logging.info("클러스터링 대상 데이터가 없습니다.")
        return []
    
    # 임베딩 배열 생성 및 클러스터링 수행
    X = np.array([item["embedding"] for item in items])
    logging.debug("클러스터링 대상 임베딩 shape: %s", X.shape)
    
    # HDBSCAN으로 클러스터링 수행
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    clusterer.fit(X)
    raw_labels = clusterer.labels_
    logging.debug("HDBSCAN 원래 클러스터 라벨: %s", raw_labels)
    
    # 클러스터 그룹별로 묶기 (기존 HDBSCAN ID 사용)
    clusters_by_raw_id = {}
    for item, label in zip(items, raw_labels):
        clusters_by_raw_id.setdefault(label, []).append(item)
    
    # 안정적인 순서를 위해, 각 클러스터의 최소 촬영 시각 기준으로 정렬하고, 새 번호(0,1,2,…) 할당
    sorted_clusters = sorted(
        clusters_by_raw_id.items(),
        key=lambda pair: min(item["timestamp"] for item in pair[1])
    )
    
    stable_clusters = {}
    for new_id, (raw_id, cluster_items) in enumerate(sorted_clusters):
        for item in cluster_items:
            item["cluster_id"] = new_id
            # 저장된 클러스터 라벨 매핑 적용
            mapping = load_cluster_labels()
            if str(new_id) in mapping:
                item["cluster_label"] = mapping[str(new_id)]
            else:
                item["cluster_label"] = str(new_id)
        stable_clusters[new_id] = cluster_items
        logging.debug("안정적 클러스터 %d에 할당된 이미지: %s", new_id, [it["filename"] for it in cluster_items])
    
    # 최종적으로 stable_clusters.values()를 리스트로 변환하여 반환
    final_clusters = list(stable_clusters.values())
    logging.info("총 %d 개의 안정적 클러스터 생성", len(final_clusters))
    return final_clusters
#############################################
# 클러스터 라벨 업데이트 엔드포인트
#############################################
@app.route('/update_cluster_label', methods=['POST'])
def update_cluster_label():
    data = request.get_json()
    cluster_id = data.get("cluster_id")
    cluster_label = data.get("cluster_label")
    if cluster_id is None or cluster_label is None:
        return jsonify({"error": "cluster_id와 cluster_label은 필수입니다."}), 400
    mapping = load_cluster_labels()
    mapping[str(cluster_id)] = cluster_label
    save_cluster_labels(mapping)
    logging.info("클러스터 %s에 라벨 '%s' 업데이트됨", cluster_id, cluster_label)
    return jsonify({"message": "클러스터 라벨 업데이트 완료", "cluster_id": cluster_id, "cluster_label": cluster_label}), 200

#############################################
# /upload 엔드포인트: 파일 업로드 및 추론 실행
#############################################
@app.route('/upload', methods=['POST'])
def upload_file():
    global LAST_UPLOAD_TIME
    if 'file' not in request.files:
        return jsonify({'error': '파일 파트가 없습니다.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '허용되지 않은 파일 형식입니다.'}), 400

    caption = request.form.get("caption")
    file_bytes = file.read()
    try:
        image_temp = Image.open(BytesIO(file_bytes))
        exif = image_temp._getexif()
        capture_time = None
        if exif:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == "DateTimeOriginal":
                    datetime_str = value
                    date_part, time_part = datetime_str.split(" ")
                    date_part = date_part.replace(":", "-")
                    capture_time = datetime.datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
                    break
        if not capture_time:
            capture_time = datetime.datetime.now()
    except Exception as e:
        logging.error("EXIF 추출 오류: %s", e)
        capture_time = datetime.datetime.now()
    
    folder_name = capture_time.strftime("%Y-%m-%d")
    target_dir = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        logging.info("새 폴더 생성: %s", target_dir)
    filename = secure_filename(f"{capture_time.strftime('%H-%M-%S')}_{file.filename}")
    file_path = os.path.join(target_dir, filename)
    with open(file_path, 'wb') as f:
        f.write(file_bytes)
    logging.info("파일 저장 완료: %s", file_path)
    
    inference_result = run_all_inferences(file_path)
    if inference_result.get("face", {}).get("face_count", 0) > 0 and caption:
        inference_result["face"]["user_label"] = caption
    
    json_path = file_path.rsplit('.', 1)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(inference_result, jf, ensure_ascii=False, indent=4)
    logging.info("추론 결과 저장 완료: %s", json_path)
    
    LAST_UPLOAD_TIME = datetime.datetime.now()
    logging.info("[%s] 추론 완료, 총 소요 시간: %.3f초", LAST_UPLOAD_TIME, inference_result["total_inference_time"])
    
    return jsonify({
        'message': '파일 업로드 및 추론 실행 완료',
        'file_path': file_path,
        'inference': inference_result
    }), 200

#############################################
# /reset_inference 엔드포인트: 전체 이미지 재추론 (재추론 버튼 전용)
#############################################
@app.route('/reset_inference', methods=['GET'])
def reset_inference():
    global LAST_UPLOAD_TIME
    total_images = 0
    processed = 0
    logging.info("전체 재추론 시작...")
    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if allowed_file(file):
                    total_images += 1
    logging.info("총 %d 개의 이미지 발견", total_images)
    
    # 기존 JSON 파일 삭제 후 재처리
    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if allowed_file(file):
                    base_filename = file.rsplit('.', 1)[0]
                    json_filepath = os.path.join(folder_path, base_filename + ".json")
                    file_path = os.path.join(folder_path, file)
                    if os.path.exists(json_filepath):
                        os.remove(json_filepath)
                        logging.info("삭제됨: %s", json_filepath)
                    inference_result = run_all_inferences(file_path)
                    with open(json_filepath, "w", encoding="utf-8") as jf:
                        json.dump(inference_result, jf, ensure_ascii=False, indent=4)
                    processed += 1
                    percent = (processed / total_images) * 100
                    logging.info("재추론 진행: %d/%d (%.1f%%) - %s", processed, total_images, percent, file_path)
    
    LAST_UPLOAD_TIME = datetime.datetime.now()
    logging.info("[%s] 전체 재추론 완료, 처리된 이미지: %d", LAST_UPLOAD_TIME, total_images)
    
    flash("전체 재추론이 완료되었습니다.", "success")
    # 클러스터링 결과 캐시를 갱신
    clusters = cluster_face_images()
    with open(CLUSTER_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=4)
    return redirect(url_for('index'))

#############################################
# /last_upload 엔드포인트: 마지막 업로드(추론 완료) 시간 반환 (AJAX용)
#############################################
@app.route('/last_upload', methods=['GET'])
def last_upload():
    global LAST_UPLOAD_TIME
    if LAST_UPLOAD_TIME:
        return jsonify({"last_upload": LAST_UPLOAD_TIME.isoformat()})
    else:
        return jsonify({"last_upload": ""})

#############################################
# /face_cluster/<int:cluster_id> 엔드포인트: 인물 클러스터 상세보기
#############################################
@app.route('/face_cluster/<int:cluster_id>', methods=['GET'])
def face_cluster(cluster_id):
    clusters = cluster_face_images()
    # clusters는 리스트의 리스트 형태
    try:
        cluster = clusters[cluster_id]
    except IndexError:
        return "Invalid cluster id", 404
    return render_template('face_cluster.html', cluster=cluster, cluster_id=cluster_id)

#############################################
# /data/<folder>/<filename> 엔드포인트: 이미지 서빙
#############################################
@app.route('/data/<folder>/<filename>')
def serve_image(folder, filename):
    folder_path = os.path.join(DATA_DIR, folder)
    return send_from_directory(folder_path, filename)

#############################################
# / 엔드포인트: 웹 뷰어 - 그룹별 이미지 확인
#############################################
@app.route('/')
def index():
    group = request.args.get("group")
    image_list = []
    if group == "face":
        clusters = cluster_face_images()
        # index 화면에서는 각 클러스터의 대표 항목(첫 번째 이미지)을 사용
        for idx, cluster in enumerate(clusters):
            rep = cluster[0]
            rep["cluster_id"] = idx
            image_list.append(rep)
    elif group and group != "전체":
        for folder in os.listdir(DATA_DIR):
            folder_path = os.path.join(DATA_DIR, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if allowed_file(file):
                        base_filename = file.rsplit('.', 1)[0]
                        json_filepath = os.path.join(folder_path, base_filename + ".json")
                        inference_info = None
                        if os.path.exists(json_filepath):
                            with open(json_filepath, "r", encoding="utf-8") as jf:
                                inference_info = json.load(jf)
                        if group == "hand" and inference_info and inference_info.get("hand", {}).get("hand_count", 0) > 0:
                            image_list.append({'folder': folder, 'filename': file,
                                               'filepath': f"/data/{folder}/{file}", 'inference': inference_info})
                        elif group == "body" and inference_info and inference_info.get("body", {}).get("landmark_count", 0) >= 15:
                            image_list.append({'folder': folder, 'filename': file,
                                               'filepath': f"/data/{folder}/{file}", 'inference': inference_info})
                        elif group == "skin" and inference_info and inference_info.get("skin", {}).get("predicted_label"):
                            image_list.append({'folder': folder, 'filename': file,
                                               'filepath': f"/data/{folder}/{file}", 'inference': inference_info})
                        elif group == "uncategorized" and (not inference_info or inference_info.get("task") == "uncategorized"):
                            image_list.append({'folder': folder, 'filename': file,
                                               'filepath': f"/data/{folder}/{file}", 'inference': inference_info})
    else:
        temp = []
        for folder in os.listdir(DATA_DIR):
            folder_path = os.path.join(DATA_DIR, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if allowed_file(file):
                        base_filename = file.rsplit('.', 1)[0]
                        json_filepath = os.path.join(folder_path, base_filename + ".json")
                        inference_info = None
                        if os.path.exists(json_filepath):
                            with open(json_filepath, "r", encoding="utf-8") as jf:
                                inference_info = json.load(jf)
                        temp.append({'folder': folder, 'filename': file,
                                     'filepath': f"/data/{folder}/{file}", 'inference': inference_info})
        image_list = temp
    image_list.sort(key=lambda x: (x['folder'], x['filename']))
    return render_template('index.html', images=image_list, group=group)

#############################################
# Flask 서버 실행 및 mDNS 광고 동시에 실행
#############################################
if __name__ == '__main__':
    # mDNS 광고를 별도의 데몬 스레드에서 실행
    mdns_thread = threading.Thread(target=register_mdns_service, daemon=True)
    mdns_thread.start()

    # Flask 서버 실행
    app.run(host='0.0.0.0', port=8000, debug=True)