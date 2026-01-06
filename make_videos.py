
import cv2
import os
import glob
import numpy as np

def resize_with_padding(image, target_size):
    """
    이미지 비율을 유지하며 target_size(width, height)의 중앙에 배치하고,
    나머지 공간은 검은색으로 채웁니다.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas

def images_to_video(
    input_root="outputs/botsort/person",
    output_root="outputs/person_videos",
    fps=30  # 끊김 방지를 위해 채움 로직 적용
):
    """
    저장된 모든 ID 폴더를 순회하며 영상을 생성합니다.
    - 조건: 각 ID의 마지막 프레임 기준 (Max-166) ~ (Max-66) 구간
    - 기능: 빈 프레임은 직전 프레임으로 채워(Interpolation) 부드럽게 만듦
    """
    os.makedirs(output_root, exist_ok=True)

    if not os.path.exists(input_root):
        print(f"[오류] 경로 없음: {input_root}")
        return

    id_folders = sorted(os.listdir(input_root))
    print(f"=== 영상 변환 시작 (대상: 모든 감지된 ID) ===")
    
    count = 0

    for folder_name in id_folders:
        person_dir = os.path.join(input_root, folder_name)
        if not os.path.isdir(person_dir):
            continue

        # 1. 파일 파싱
        all_files = glob.glob(os.path.join(person_dir, "*.jpg"))
        if not all_files:
            continue

        frame_data = []
        for f_path in all_files:
            filename = os.path.basename(f_path)
            try:
                frame_num = int(filename.split('.')[0])
                frame_data.append((frame_num, f_path))
            except ValueError:
                continue
        
        if not frame_data:
            continue

        # 2. 구간 계산 (각 ID별 마지막 시점 기준)
        frame_data.sort(key=lambda x: x[0])
        max_frame = frame_data[-1][0]
        
        target_start = max_frame - 140
        target_end = max_frame - 40
        
        # 딕셔너리로 변환 (빠른 검색용)
        # 해당 구간(target_start ~ target_end)에 있는 파일만 필터링
        img_dict = {}
        for f_num, f_path in frame_data:
            if target_start <= f_num <= target_end:
                img_dict[f_num] = f_path
        
        # 구간 내에 이미지가 하나도 없으면 영상 생성 스킵
        if not img_dict:
            # print(f"[Skip] {folder_name}: 지정된 구간({target_start}~{target_end})에 데이터 없음.")
            continue

        # 3. 캔버스 크기 결정 (이 ID의 해당 구간 이미지들 기준)
        max_w, max_h = 0, 0
        sample_paths = list(img_dict.values())
        
        for p in sample_paths:
            img = cv2.imread(p)
            if img is None: continue
            h, w = img.shape[:2]
            if w > max_w: max_w = w
            if h > max_h: max_h = h
        
        if max_w == 0 or max_h == 0: continue

        # 짝수 맞춤 (인코딩 호환성)
        if max_w % 2 != 0: max_w += 1
        if max_h % 2 != 0: max_h += 1
        canvas_size = (max_w, max_h)

        # 4. 비디오 생성 (프레임 채우기 로직 포함)
        save_path = os.path.join(output_root, f"{folder_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, canvas_size)

        last_valid_img = None
        
        # start ~ end 까지 빈틈없이 순회
        for f_idx in range(target_start, target_end + 1):
            if f_idx in img_dict:
                # 파일이 있으면 읽어서 기록 & 갱신
                img = cv2.imread(img_dict[f_idx])
                if img is not None:
                    frame_final = resize_with_padding(img, canvas_size)
                    out.write(frame_final)
                    last_valid_img = frame_final
            else:
                # 파일이 없으면 직전 프레임 재사용 (부드러운 재생)
                if last_valid_img is not None:
                    out.write(last_valid_img)
                else:
                    # 시작부터 데이터가 없으면 검은 화면
                    blank = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
                    out.write(blank)

        out.release()
        count += 1
        print(f"[생성 완료] {folder_name}.mp4 (구간: {target_start}~{target_end}, 파일 수: {len(img_dict)})")

    print("="*40)
    print(f"총 {count}개의 비디오 생성 완료.")

if __name__ == "__main__":
    # 단독 테스트용
    images_to_video()