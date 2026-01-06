import infinf as inf
import media_csv_v2 as medcsv
import mediapipe_test as medpip
import crop_to_video as c_to_v

if __name__ == "__main__":
    c_to_v.images_to_video(
        input_root="outputs/botsort/person",
        output_root="outputs/person_videos",
        target_start=550,
        target_end=650,
        fps=10
    )
    medpip.main()
    medcsv.main()
    inf.main()

