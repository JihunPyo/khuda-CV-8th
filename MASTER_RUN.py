import sys
import os
import glob
import torch
from pathlib import Path

# ---------------------------------------------------------
# 모듈 임포트
# (주의: 아래 모듈들은 경로를 인자로 받도록 수정된 버전이어야 합니다)
# ---------------------------------------------------------
from tracker import run_track_and_save
from roi_pick import get_roi_points
from make_videos import images_to_video
import mediapipe_test as medpip
import media_csv_v2 as medcsv
import infinf as inf

# =========================================================
# [설정] 경로 및 파라미터
# =========================================================
# 1. 입력 영상
VIDEO_FILE = "inputs/normal_test2.mp4"

# 2. 모델 경로
YOLO_MODEL = "yolo11n.pt"
TRACKER_YAML = "botsort.yaml"
LSTM_CKPT = Path("inference/epoch_015.pt") # LSTM 모델 위치

# 3. 출력 경로 설정
BASE_OUT_DIR = "outputs"
BOTSORT_DIR = os.path.join(BASE_OUT_DIR, "botsort")
PERSON_CROP_DIR = os.path.join(BOTSORT_DIR, "person")
VIDEO_OUT_DIR = os.path.join(BASE_OUT_DIR, "person_videos")
INFERENCE_OUT_DIR = os.path.join(BASE_OUT_DIR, "inference")

def main():
    # # -----------------------------------------------------
    # # Step 1: ROI 설정 및 객체 추적 (YOLO + BotSORT)
    # # -----------------------------------------------------
    # print("\n" + "="*60)
    # print(">>> [Step 1] ROI 설정 및 객체 추적 시작")
    # print("="*60)

    # # ROI 선택
    # print(">>> ROI 선택 창을 엽니다...")
    # selected_points = get_roi_points(VIDEO_FILE)

    # if len(selected_points) < 3:
    #     print("\n[경고] 점이 3개 미만입니다. ROI 설정 없이 진행하려면 엔터.")
    
    # print(">>> YOLO AI 모델 로딩 및 분석 시작...")
    
    # # 트래킹 실행
    # run_track_and_save(
    #     video_path=VIDEO_FILE,      
    #     out_dir=BOTSORT_DIR,  
    #     roi_points=selected_points, 
        
    #     tracker_yaml=TRACKER_YAML,
    #     model_path=YOLO_MODEL,
    #     conf=0.10,                  
    #     imgsz=1280,                 
    #     save_vis=True,              
    #     save_bbox_csv=True,         
    #     bbox_only_person=True,      
    #     save_crops=True,            

    #     enable_id_fix=True,         
    #     idfix_max_age=45,           
    #     idfix_use_appearance=True,
    #     crop_padding=0.5 
    # )

    # # -----------------------------------------------------
    # # Step 2: 크롭 이미지 -> 영상 변환
    # # -----------------------------------------------------
    # print("\n" + "="*60)
    # print(">>> [Step 2] 추적된 객체(사람)를 영상으로 변환")
    # print("="*60)
    
    # # images_to_video 함수가 outputs/person_videos 에 .mp4 생성
    # images_to_video(
    #     input_root=PERSON_CROP_DIR,
    #     output_root=VIDEO_OUT_DIR,
    #     fps=30
    # )

    # -----------------------------------------------------
    # Step 3: 이상 행동 추론 (MediaPipe -> CSV -> LSTM)
    # -----------------------------------------------------
    print("\n" + "="*60)
    print(">>> [Step 3] 이상 행동(Normal/Abnormal) 추론 시작")
    print("="*60)

    # 3-1. 생성된 비디오 목록 가져오기
    video_files = glob.glob(os.path.join(VIDEO_OUT_DIR, "*.mp4"))
    if not video_files:
        print("[알림] 생성된 사람 영상이 없습니다. 추론을 종료합니다.")
        return

    # 3-2. LSTM 모델 로드 (1회)
    device = torch.device("mps")
    print(f">>> LSTM 모델 로딩 중... (Device: {device})")
    
    try:
        model, model_cfg = inf.load_model(LSTM_CKPT, device)
    except Exception as e:
        print(f"[오류] 모델 로드 실패: {e}")
        return

    # 3-3. 각 비디오별 처리 루프
    results = []
    os.makedirs(INFERENCE_OUT_DIR, exist_ok=True)

    for v_path in sorted(video_files):
        filename = os.path.basename(v_path)
        file_id = os.path.splitext(filename)[0] # 예: id_0001
        
        print(f"\n--- Processing: {file_id} ---")
        
        # 파일 경로 정의
        npz_path = os.path.join(INFERENCE_OUT_DIR, f"{file_id}.npz")
        csv_path = os.path.join(INFERENCE_OUT_DIR, f"{file_id}.csv")

        # (A) MediaPipe 추출 (Video -> NPZ)
        if not medpip.process_video(v_path, npz_path):
            print(f"   -> [Skip] MediaPipe 추출 실패")
            continue

        # (B) CSV 변환 및 전처리 (NPZ -> CSV)
        if not medcsv.process_npz_to_csv(npz_path, csv_path):
            print(f"   -> [Skip] CSV 변환 실패 (데이터 부족 등)")
            continue

        # (C) 추론 (CSV -> Result)
        try:
            res = inf.predict_single(model, csv_path, device, model_cfg)
            
            status = "ABNORMAL 🚨" if res['pred_class'] == 1 else "Normal ✅"
            prob = res['p_abnormal'] if res['pred_class'] == 1 else res['p_normal']
            
            print(f"   -> 결과: {status} ({prob:.2%})")
            
            results.append({
                "id": file_id,
                "status": status,
                "prob": prob
            })
            
        except Exception as e:
            print(f"   -> [오류] 추론 중 에러 발생: {e}")

    # -----------------------------------------------------
    # 최종 결과 리포트
    # -----------------------------------------------------
    print("\n" + "="*60)
    print("FINAL SUMMARY REPORT")
    print("="*60)
    if not results:
        print("검출된 유효 데이터가 없습니다.")
    else:
        for r in results:
            print(f"ID: {r['id']:<10} | {r['status']} (확률: {r['prob']:.4f})")
    print("="*60)

   # -----------------------------------------------------
    # Step 4: 최종 데모 영상 생성 (Overlay)
    # -----------------------------------------------------

    import json
    from overlay_demo import render

    # results 리스트를 JSON으로 저장(overlay_demo.py가 읽을 포맷)
    os.makedirs(INFERENCE_OUT_DIR, exist_ok=True)
    results_json_path=os.path.join(INFERENCE_OUT_DIR, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # bboxes.csv 경로(현재 tracker.py 기준)
    bboxes_csv=os.path.join(BOTSORT_DIR, "bboxes.csv")

    # 최종 데모 영상 경로
    final_demo=os.path.join(BASE_OUT_DIR, "final_demo_deco.mp4")

    render(
        video_path=VIDEO_FILE,
        bboxes_csv=bboxes_csv,
        results_json=results_json_path,
        out_path=final_demo,
        only_person=True,
        smooth=True,
    )

    print(f"[DONE] Final demo saved: {final_demo}")    

if __name__ == "__main__":
    main()