#!/usr/bin/env python3
"""
DermPhotoServer

웹 기반 이미지 분류 및 클러스터링 서버
- 업로드된 이미지는 날짜별 폴더 내의 images/에 저장되고, 메타데이터는 metadata/에 저장됨.
- HDBSCAN 기반 클러스터링 후 안정적 클러스터 번호와 라벨을 적용.
- Flask 웹서버, 파일 삭제, 재추론, 클러스터 라벨 업데이트 등 기능 제공.
- Tkinter GUI를 통한 로그 출력 및 자동 포트 선택 기능 포함 (GUI 사용 환경에 한함).

작성자: Your Name
날짜: 2025-04-09
"""

import sys, platform, os, datetime, json, time, socket, threading, logging
import cv2, numpy as np
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags

# 조건부 임포트: macOS arm64에서는 TensorFlow Lite 인터프리터 사용, 기타 환경은 tflite-runtime 사용
if sys.platform == 'darwin' and platform.machine() == 'arm64':
    from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter
else:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter

import mediapipe as mp
import hdbscan
import face_recognition
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

# 로깅 설정 (DEBUG 레벨)
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

###############################
# Flask 앱 설정 및 기본 변수
###############################
app = Flask(__name__)
app.secret_key = 'your_secret_key'
Interpreter = TFLiteInterpreter  # 편의상 사용

LAST_UPLOAD_TIME = None  # 마지막 업로드 시간

# 기본 경로 및 폴더 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
# DATA_DIR 내부 항목 중 숨김 파일(이름이 '.'로 시작하는)은 무시
for item in os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []:
    if item.startswith('.'):
        logging.debug("DATA_DIR 내부 숨김 항목 무시: %s", item)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    logging.info("DATA_DIR 생성: %s", DATA_DIR)

# 하위 폴더 이름
IMAGES_SUBDIR = "images"
METADATA_SUBDIR = "metadata"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 클러스터 라벨 매핑 파일 및 결과 캐시 파일 경로
CLUSTER_LABELS_FILE = os.path.join(BASE_DIR, "cluster_labels.json")
CLUSTER_RESULTS_FILE = os.path.join(BASE_DIR, "cluster_results.json")

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

def load_labels_txt(txt_file_path):
    with open(txt_file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

LABELS_FILE = os.path.join(BASE_DIR, "model", "skin", "labels_skin_cancer.txt")
labels_skin = load_labels_txt(LABELS_FILE)
logging.info("Loaded Skin Cancer Labels: %s", labels_skin)

KOREAN_LABEL_MAP = {
    "akiec": "광선각화증 및 상피내암",
    "bcc": "기저세포암",
    "bkl": "양성 각화증 유사 병변",
    "df": "피부 섬유종",
    "mel": "흑색종",
    "nv": "멜라닌 모반",
    "vasc": "혈관 병변"
}

# TFLite 피부 질환 분류 모델 로드
SKIN_MODEL_PATH = os.path.join(BASE_DIR, "model", "skin", "skin_cancer_best_model.tflite")
interpreter_skin = Interpreter(model_path=SKIN_MODEL_PATH)
interpreter_skin.allocate_tensors()
skin_input_details = interpreter_skin.get_input_details()
skin_output_details = interpreter_skin.get_output_details()
logging.info("TFLite 모델 로드 완료: %s", SKIN_MODEL_PATH)

###############################
# 추론 함수들
###############################
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

def normalize_embedding(embedding):
    arr = np.array(embedding)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm != 0 else arr.tolist()

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

def get_timestamp_from_filename(folder, filename):
    try:
        folder_date = datetime.datetime.strptime(folder, "%Y-%m-%d").date()
    except Exception:
        folder_date = datetime.datetime.now().date()
    current_time = datetime.datetime.now().time()
    return datetime.datetime.combine(folder_date, current_time)

###############################################
# 얼굴 클러스터링 함수 (HDBSCAN 및 안정적 클러스터 번호, 라벨 적용)
###############################################
def cluster_face_images():
    items = []
    logging.debug("클러스터링 함수 시작")
    # DATA_DIR의 항목 중 숨김 파일은 건너뜁니다.
    for date_dir in os.listdir(DATA_DIR):
        if date_dir.startswith('.'):
            continue
        date_path = os.path.join(DATA_DIR, date_dir)
        if not os.path.isdir(date_path):
            continue
        images_path = os.path.join(date_path, IMAGES_SUBDIR)
        metadata_path = os.path.join(date_path, METADATA_SUBDIR)
        if not (os.path.isdir(images_path) and os.path.isdir(metadata_path)):
            continue
        for file in os.listdir(images_path):
            if allowed_file(file):
                base_filename = file.rsplit('.', 1)[0]
                json_filepath = os.path.join(metadata_path, base_filename + ".json")
                if os.path.exists(json_filepath):
                    with open(json_filepath, "r", encoding="utf-8") as jf:
                        inference_info = json.load(jf)
                    if inference_info and inference_info.get("face", {}).get("face_count", 0) > 0:
                        ts = get_timestamp_from_filename(date_dir, file)
                        ts_str = ts.isoformat()
                        for embedding in inference_info["face"]["face_embeddings"]:
                            norm_embedding = normalize_embedding(embedding)
                            items.append({
                                "date": date_dir,
                                "filename": file,
                                "filepath": f"/data/{date_dir}/{IMAGES_SUBDIR}/{file}",
                                "inference": inference_info,
                                "timestamp": ts_str,
                                "embedding": norm_embedding
                            })
    if not items:
        logging.info("클러스터링 대상 데이터가 없습니다.")
        return []
    X = np.array([item["embedding"] for item in items])
    logging.debug("클러스터링 대상 임베딩 shape: %s", X.shape)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    clusterer.fit(X)
    raw_labels = clusterer.labels_
    logging.debug("HDBSCAN 원래 클러스터 라벨: %s", raw_labels)
    clusters_by_raw_id = {}
    for item, label in zip(items, raw_labels):
        clusters_by_raw_id.setdefault(label, []).append(item)
    sorted_clusters = sorted(
        clusters_by_raw_id.items(),
        key=lambda pair: min(item["timestamp"] for item in pair[1])
    )
    stable_clusters = {}
    for new_id, (raw_id, cluster_items) in enumerate(sorted_clusters):
        for item in cluster_items:
            item["cluster_id"] = new_id
            mapping = load_cluster_labels()
            if str(new_id) in mapping:
                item["cluster_label"] = mapping[str(new_id)]
            else:
                item["cluster_label"] = str(new_id)
        stable_clusters[new_id] = cluster_items
        logging.debug("안정적 클러스터 %d에 할당된 이미지: %s", new_id, [it["filename"] for it in cluster_items])
    final_clusters = list(stable_clusters.values())
    logging.info("총 %d 개의 안정적 클러스터 생성", len(final_clusters))
    return final_clusters

###############################################
# 클러스터 라벨 업데이트 엔드포인트
###############################################
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

###############################################
# 파일 삭제 엔드포인트 (AJAX)
###############################################
@app.route('/delete_file', methods=['POST'])
def delete_file():
    data = request.get_json()
    file_path = data.get("file_path")
    if not file_path:
        return jsonify({"error": "file_path 필수"}), 400
    rel_path = file_path.lstrip("/data/")
    abs_image_path = os.path.join(DATA_DIR, rel_path)
    date_dir = rel_path.split(os.sep)[0]
    base_filename = os.path.basename(abs_image_path).rsplit('.', 1)[0]
    abs_meta_path = os.path.join(DATA_DIR, date_dir, METADATA_SUBDIR, base_filename + ".json")
    try:
        if os.path.exists(abs_image_path):
            os.remove(abs_image_path)
            logging.info("이미지 삭제: %s", abs_image_path)
        if os.path.exists(abs_meta_path):
            os.remove(abs_meta_path)
            logging.info("메타데이터 삭제: %s", abs_meta_path)
        return jsonify({"message": "파일 삭제 완료"}), 200
    except Exception as e:
        logging.error("파일 삭제 오류: %s", e)
        return jsonify({"error": str(e)}), 500

###############################################
# /reset_inference 엔드포인트: 전체 이미지 재추론
###############################################
@app.route('/reset_inference', methods=['GET'])
def reset_inference():
    global LAST_UPLOAD_TIME
    total_images = 0
    processed = 0
    logging.info("전체 재추론 시작...")
    # 날짜별 폴더만 처리 (숨김 항목 무시)
    for date_dir in os.listdir(DATA_DIR):
        if date_dir.startswith('.'):
            continue
        date_path = os.path.join(DATA_DIR, date_dir)
        if not os.path.isdir(date_path):
            continue
        images_path = os.path.join(date_path, IMAGES_SUBDIR)
        metadata_path = os.path.join(date_path, METADATA_SUBDIR)
        if not os.path.isdir(images_path):
            continue
        # 메타데이터 폴더가 없으면 생성
        os.makedirs(metadata_path, exist_ok=True)
        for file in os.listdir(images_path):
            if allowed_file(file):
                total_images += 1
    logging.info("총 %d 개의 이미지 발견", total_images)
    
    # 날짜별 재추론: 이미지와 메타데이터 재처리
    for date_dir in os.listdir(DATA_DIR):
        if date_dir.startswith('.'):
            continue
        date_path = os.path.join(DATA_DIR, date_dir)
        if not os.path.isdir(date_path):
            continue
        images_path = os.path.join(date_path, IMAGES_SUBDIR)
        metadata_path = os.path.join(date_path, METADATA_SUBDIR)
        os.makedirs(metadata_path, exist_ok=True)
        if os.path.isdir(images_path):
            for file in os.listdir(images_path):
                if allowed_file(file):
                    base_filename = file.rsplit('.', 1)[0]
                    json_filepath = os.path.join(metadata_path, base_filename + ".json")
                    image_file_path = os.path.join(images_path, file)
                    if os.path.exists(json_filepath):
                        os.remove(json_filepath)
                        logging.info("삭제됨: %s", json_filepath)
                    inference_result = run_all_inferences(image_file_path)
                    with open(json_filepath, "w", encoding="utf-8") as jf:
                        json.dump(inference_result, jf, ensure_ascii=False, indent=4)
                    processed += 1
                    percent = (processed / total_images) * 100
                    logging.info("재추론 진행: %d/%d (%.1f%%) - %s", processed, total_images, percent, image_file_path)
    LAST_UPLOAD_TIME = datetime.datetime.now()
    logging.info("[%s] 전체 재추론 완료, 처리된 이미지: %d", LAST_UPLOAD_TIME.isoformat(), total_images)
    flash("전체 재추론이 완료되었습니다.", "success")
    clusters = cluster_face_images()
    with open(CLUSTER_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=4)
    return redirect(url_for('index'))

###############################################
# /last_upload 엔드포인트: 마지막 업로드 시간 반환 (AJAX)
###############################################
@app.route('/last_upload', methods=['GET'])
def last_upload():
    global LAST_UPLOAD_TIME
    if LAST_UPLOAD_TIME:
        return jsonify({"last_upload": LAST_UPLOAD_TIME.isoformat()})
    return jsonify({"last_upload": ""})

###############################################
# /face_cluster/<int:cluster_id> 엔드포인트: 클러스터 상세보기
###############################################
@app.route('/face_cluster/<int:cluster_id>', methods=['GET'])
def face_cluster(cluster_id):
    clusters = cluster_face_images()
    try:
        cluster = clusters[cluster_id]
    except IndexError:
        return "Invalid cluster id", 404
    return render_template('face_cluster.html', cluster=cluster, cluster_id=cluster_id)

###############################################
# / : 메인 페이지 (이미지 그룹별 보기)
###############################################
@app.route('/')
def index():
    group = request.args.get("group")
    image_list = []
    if group == "face":
        clusters = cluster_face_images()
        for idx, cluster in enumerate(clusters):
            rep = cluster[0]
            rep["cluster_id"] = idx
            image_list.append(rep)
    elif group and group != "전체":
        for date_dir in os.listdir(DATA_DIR):
            if date_dir.startswith('.'):
                continue
            date_path = os.path.join(DATA_DIR, date_dir)
            images_path = os.path.join(date_path, IMAGES_SUBDIR)
            metadata_path = os.path.join(date_path, METADATA_SUBDIR)
            if not os.path.isdir(images_path):
                continue
            for file in os.listdir(images_path):
                if allowed_file(file):
                    base_filename = file.rsplit('.', 1)[0]
                    meta_file = os.path.join(metadata_path, base_filename + ".json")
                    inference_info = None
                    if os.path.exists(meta_file):
                        with open(meta_file, "r", encoding="utf-8") as jf:
                            inference_info = json.load(jf)
                    if group == "hand" and inference_info and inference_info.get("hand", {}).get("hand_count", 0) > 0:
                        image_list.append({'date': date_dir, 'filename': file,
                                           'filepath': f"/data/{date_dir}/{IMAGES_SUBDIR}/{file}", 'inference': inference_info})
                    elif group == "body" and inference_info and inference_info.get("body", {}).get("landmark_count", 0) >= 15:
                        image_list.append({'date': date_dir, 'filename': file,
                                           'filepath': f"/data/{date_dir}/{IMAGES_SUBDIR}/{file}", 'inference': inference_info})
                    elif group == "skin" and inference_info and inference_info.get("skin", {}).get("predicted_label"):
                        image_list.append({'date': date_dir, 'filename': file,
                                           'filepath': f"/data/{date_dir}/{IMAGES_SUBDIR}/{file}", 'inference': inference_info})
                    elif group == "uncategorized" and (not inference_info or inference_info.get("task") == "uncategorized"):
                        image_list.append({'date': date_dir, 'filename': file,
                                           'filepath': f"/data/{date_dir}/{IMAGES_SUBDIR}/{file}", 'inference': inference_info})
    else:
        for date_dir in os.listdir(DATA_DIR):
            if date_dir.startswith('.'):
                continue
            date_path = os.path.join(DATA_DIR, date_dir)
            images_path = os.path.join(date_path, IMAGES_SUBDIR)
            metadata_path = os.path.join(date_path, METADATA_SUBDIR)
            if not os.path.isdir(images_path):
                continue
            for file in os.listdir(images_path):
                if allowed_file(file):
                    base_filename = file.rsplit('.', 1)[0]
                    meta_file = os.path.join(metadata_path, base_filename + ".json")
                    inference_info = None
                    if os.path.exists(meta_file):
                        with open(meta_file, "r", encoding="utf-8") as jf:
                            inference_info = json.load(jf)
                    image_list.append({'date': date_dir, 'filename': file,
                                       'filepath': f"/data/{date_dir}/{IMAGES_SUBDIR}/{file}", 'inference': inference_info})
    image_list.sort(key=lambda x: (x.get('date', ''), x['filename']))
    return render_template('index.html', images=image_list, group=group)

###############################
# Tkinter GUI 로그 핸들러 클래스
###############################
class TkinterLogHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)
        self.text_widget.after(0, append)

###############################
# 자동 포트 선택 함수
###############################
def get_free_port(start=8000, end=8100):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                logging.debug("사용 가능한 포트 발견: %d", port)
                return port
            except OSError:
                continue
    raise RuntimeError("사용 가능한 포트 없음")

###############################
# GUI & 서버 실행 함수 (Tkinter 기반)
###############################
def start_gui_and_server():
    global PORT
    PORT = get_free_port(8000, 8100)
    server_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False))
    server_thread.setDaemon(True)
    server_thread.start()
    logging.info("Flask 서버 실행 중... (포트 %d)", PORT)
    root = tk.Tk()
    root.title(f"DermPhotoServer - 서버 실행 중: http://localhost:{PORT}")
    st = ScrolledText(root, state='disabled', height=30, width=100)
    st.pack(expand=True, fill='both')
    gui_handler = TkinterLogHandler(st)
    gui_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    gui_handler.setFormatter(formatter)
    logging.getLogger().addHandler(gui_handler)
    root.mainloop()

###############################
# 메인 실행부
###############################
if __name__ == '__main__':
    try:
        import tkinter as tk
        test_root = tk.Tk()
        test_root.withdraw()
        test_root.destroy()
        has_tkinter = True
    except Exception as e:
        logging.error("Tkinter GUI 사용 불가: %s", e)
        has_tkinter = False
    if has_tkinter:
        start_gui_and_server()
    else:
        PORT = get_free_port(8000, 8100)
        logging.info("GUI 미지원 - CLI 모드로 서버 실행: 포트 %d", PORT)
        app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)