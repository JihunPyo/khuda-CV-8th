# overlay_demo.py
# 원본 영상 위에 bbox(+track id) + LSTM 이진분류 결과를 오버레이해서 mp4로 저장
#
# 사용 예)
#   python overlay_demo.py \
#     --video inputs/normal_test2.mp4 \
#     --bboxes outputs/botsort/bboxes.csv \
#     --results outputs/inference/results.json \
#     --out outputs/final_demo.mp4
#
# 주의:
# - tracker.py에서 frame_idx가 1부터 시작하므로(bboxes.csv frame 컬럼),
#   비디오 프레임(0-index)과 매칭할 때는 frame_csv=frame_video_idx+1로 맞춥니다.

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd


def _id_to_key(stable_id:int)->str:
    return f"id_{stable_id:04d}"


def _load_results(path:str)->dict:
    """
    results.json 포맷(권장):
    [
      {"id":"id_0001","pred_class":0,"p_normal":0.91,"p_abnormal":0.09},
      {"id":"id_0002","pred_class":1,"p_normal":0.12,"p_abnormal":0.88}
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data=json.load(f)
    out={}
    if isinstance(data, dict):
        # { "id_0001": {...} } 형태도 허용
        for k, v in data.items():
            out[str(k)]=v
        return out
    for r in data:
        out[str(r.get("id"))]=r
    return out


def _analysis_windows_from_csv(df:pd.DataFrame, start_offset:int=140, end_offset:int=40)->dict:
    """
    make_videos.py와 동일하게 max_frame 기준으로 분석 구간을 잡아줌.
    return: {stable_id:(start_frame,end_frame,max_frame)}
    """
    win={}
    g=df.groupby("stable_id")["frame"].max()
    for sid, maxf in g.items():
        maxf=int(maxf)
        win[int(sid)]=(maxf-start_offset, maxf-end_offset, maxf)
    return win


def _draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, corner=18):
    x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
    # 좌상
    cv2.line(img, (x1, y1), (x1+corner, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1+corner), color, thickness)
    # 우상
    cv2.line(img, (x2, y1), (x2-corner, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1+corner), color, thickness)
    # 좌하
    cv2.line(img, (x1, y2), (x1+corner, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2-corner), color, thickness)
    # 우하
    cv2.line(img, (x2, y2), (x2-corner, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2-corner), color, thickness)


def _alpha_rect(img, x1, y1, x2, y2, color, alpha=0.55):
    x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
    x1=max(0, x1); y1=max(0, y1)
    x2=min(img.shape[1]-1, x2); y2=min(img.shape[0]-1, y2)
    if x2<=x1 or y2<=y1:
        return
    overlay=img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)


def _put_label(img, x, y, title, subtitle=None, color_bg=(0,0,0), color_fg=(255,255,255), progress=None):
    """
    progress: 0~1이면 하단에 progress bar 표시
    """
    font=cv2.FONT_HERSHEY_SIMPLEX
    scale=0.62
    thickness=2

    (tw, th), _=cv2.getTextSize(title, font, scale, thickness)
    subw=subh=0
    if subtitle:
        (subw, subh), _=cv2.getTextSize(subtitle, font, 0.52, 1)

    pad=8
    w=max(tw, subw)+pad*2
    h=th+pad*2+(subh+6 if subtitle else 0)+(10 if progress is not None else 0)

    x1=x
    y1=y-h
    x2=x+w
    y2=y

    _alpha_rect(img, x1, y1, x2, y2, color_bg, alpha=0.58)
    cv2.putText(img, title, (x1+pad, y1+pad+th), font, scale, color_fg, thickness, cv2.LINE_AA)
    if subtitle:
        cv2.putText(img, subtitle, (x1+pad, y1+pad+th+subh+6), font, 0.52, color_fg, 1, cv2.LINE_AA)

    if progress is not None:
        px1=x1+pad
        px2=x2-pad
        py2=y2-pad
        py1=py2-6
        cv2.rectangle(img, (px1, py1), (px2, py2), (255,255,255), 1)
        fill=int(px1+(px2-px1)*float(np.clip(progress, 0.0, 1.0)))
        cv2.rectangle(img, (px1, py1), (fill, py2), (255,255,255), -1)


def _ema(prev, cur, a=0.75):
    if prev is None:
        return cur
    return (a*np.array(prev)+(1-a)*np.array(cur)).tolist()


def render(video_path:str, bboxes_csv:str, results_json:str, out_path:str, only_person=True, smooth=True):
    df=pd.read_csv(bboxes_csv)
    if only_person:
        df=df[df["class"]=="person"].copy()

    # (x,y,w,h)->(x1,y1,x2,y2)
    df["x2"]=df["x"]+df["w"]
    df["y2"]=df["y"]+df["h"]

    # frame별로 빠르게 접근
    by_frame=defaultdict(list)
    for r in df.itertuples(index=False):
        by_frame[int(r.frame)].append({
            "stable_id":int(r.stable_id),
            "x1":int(r.x),
            "y1":int(r.y),
            "x2":int(r.x2),
            "y2":int(r.y2),
            "conf":float(r.conf),
        })

    results=_load_results(results_json)
    windows=_analysis_windows_from_csv(df)

    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0:
        fps=30.0
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    out=cv2.VideoWriter(out_path, fourcc, fps, (W,H))

    # bbox 스무딩용
    smooth_bbox={}
    last_seen={}  # sid->frame

    vid_idx=0
    while True:
        ok, frame=cap.read()
        if not ok:
            break
        vid_idx+=1
        frame_csv=vid_idx  # tracker.csv는 1부터

        dets=by_frame.get(frame_csv, [])
        # HUD
        hud=f"Frame {frame_csv}/{nframes}"
        _put_label(frame, 12, 44, hud, subtitle=f"Tracks on frame: {len(dets)}", color_bg=(0,0,0))

        for d in dets:
            sid=d["stable_id"]
            key=_id_to_key(sid)

            x1,y1,x2,y2=d["x1"], d["y1"], d["x2"], d["y2"]
            if smooth:
                prev=smooth_bbox.get(sid)
                sm=_ema(prev, [x1,y1,x2,y2], a=0.78)
                smooth_bbox[sid]=sm
                x1,y1,x2,y2=map(int, sm)

            last_seen[sid]=frame_csv

            # 결과 조회
            rr=results.get(key)
            if rr is None:
                label="UNKNOWN"
                p=None
                is_ab=None
            else:
                is_ab=int(rr.get("pred_class", 0))==1
                p=float(rr.get("p_abnormal", 0.0))
                label="ABNORMAL" if is_ab else "NORMAL"

            # 분석 진행도(‘기깔’ 요소)
            prog=None
            if sid in windows:
                s,e,_=windows[sid]
                if e>s:
                    prog=(frame_csv-s)/float(e-s)

            # 색상 (BGR)
            if is_ab is None:
                color=(0,255,255)
                bg=(0,0,0)
            elif is_ab:
                color=(0,0,255)
                bg=(0,0,80)
            else:
                color=(0,255,0)
                bg=(0,60,0)

            _draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, corner=18)

            title=f"{key}  {label}"
            subtitle=f"P(abnormal)={p:.2f}" if p is not None else "no inference"
            # bbox 위에 패널 배치
            py=max(22, y1-6)
            _put_label(frame, x1, py, title, subtitle=subtitle, color_bg=bg, progress=prog)

        # 오래 안 보인 bbox 스무딩 상태 정리
        if smooth and (vid_idx%30==0):
            dead=[]
            for sid, lastf in last_seen.items():
                if frame_csv-lastf>60:
                    dead.append(sid)
            for sid in dead:
                last_seen.pop(sid, None)
                smooth_bbox.pop(sid, None)

        out.write(frame)

    cap.release()
    out.release()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--bboxes", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="person만이 아니라 모든 class bbox를 그리고 싶으면 사용")
    ap.add_argument("--no_smooth", action="store_true")
    args=ap.parse_args()

    render(
        video_path=args.video,
        bboxes_csv=args.bboxes,
        results_json=args.results,
        out_path=args.out,
        only_person=(not args.all),
        smooth=(not args.no_smooth),
    )


if __name__=="__main__":
    main()
