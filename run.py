# # run.py
# import sys
# import time
# from tracker import run_track_and_save
# from roi_pick import get_roi_points 

# def main():
#     video_file = "inputs/test7.mp4"

#     # 1. ROI 좌표 선택
#     print(">>> ROI 선택 창을 엽니다...")
#     selected_points = get_roi_points(video_file)

#     if len(selected_points) < 3:
#         print("\n[경고] 점이 3개 미만입니다. ROI 설정 없이 진행하려면 엔터, 종료하려면 Ctrl+C")
#         # 필요 시 여기서 sys.exit() 처리 가능
    
#     # 2. 안내 메시지 (사용자가 멈췄다고 생각하지 않게)
#     print("\n" + "="*50)
#     print(">>> [시스템] ROI 설정 완료.")
#     print(">>> [시스템] YOLO AI 모델을 메모리에 로딩 중입니다... (약 5~10초 소요)")
#     print(">>> [시스템] 로딩이 끝나면 분석이 시작됩니다. 잠시만 기다려주세요.")
#     print("="*50 + "\n")

#     # 3. 트래킹 실행
#     run_track_and_save(
#         video_path=video_file,      
#         out_dir="outputs/botsort",  
#         roi_points=selected_points, 
        
#         tracker_yaml="botsort.yaml",
#         model_path="yolo11n.pt",
#         conf=0.10,                  
#         imgsz=1280,                 
#         save_vis=True,              
#         save_bbox_csv=True,         
#         bbox_only_person=True,      
#         save_crops=True,            

#         enable_id_fix=True,         
#         idfix_max_age=45,           
#         idfix_use_appearance=True,  
#     )

# if __name__ == "__main__":
#     main()




# run.py
import sys
import os
import time
from tracker import run_track_and_save
from roi_pick import get_roi_points
from make_videos import images_to_video  # [

def main():
    video_file = "inputs/normal_test1.mp4"
    
    # 출력 경로 설정 (일관성을 위해 변수화)
    base_output_dir = "outputs/botsort"
    person_crop_dir = os.path.join(base_output_dir, "person")
    video_output_dir = "outputs/person_videos"

    # 1. ROI 좌표 선택
    print(">>> ROI 선택 창을 엽니다...")
    selected_points = get_roi_points(video_file)

    if len(selected_points) < 3:
        print("\n[경고] 점이 3개 미만입니다. ROI 설정 없이 진행하려면 엔터, 종료하려면 Ctrl+C")
    
    # 2. 안내 메시지
    print("\n" + "="*50)
    print(">>> [시스템] ROI 설정 완료.")
    print(">>> [시스템] YOLO AI 모델을 메모리에 로딩 중입니다... (약 5~10초 소요)")
    print(">>> [시스템] 로딩이 끝나면 분석이 시작됩니다.")
    print("="*50 + "\n")

    # 3. 트래킹 및 크롭 이미지 저장 실행
    run_track_and_save(
        video_path=video_file,      
        out_dir=base_output_dir,  
        roi_points=selected_points, 
        
        tracker_yaml="botsort.yaml",
        model_path="yolo11n.pt",
        conf=0.10,                  
        imgsz=1280,                 
        save_vis=True,              
        save_bbox_csv=True,         
        bbox_only_person=True,      
        save_crops=True,            

        enable_id_fix=True,         
        idfix_max_age=45,           
        idfix_use_appearance=True,
        crop_padding=0.5 # tracker.py v2.0 기능 (여유 공간)
    )
    
    # 4. [NEW] 결과 이미지를 영상으로 변환 (조건: Last-166 ~ Last-66)
    print("\n" + "="*50)
    print(">>> [시스템] 트래킹 완료. 결과 영상을 생성합니다...")
    print(f">>> 대상 폴더: {person_crop_dir}")
    print("="*50 + "\n")
    
    images_to_video(
        input_root=person_crop_dir,
        output_root=video_output_dir,
        fps=30
    )

    print("\n>>> 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()