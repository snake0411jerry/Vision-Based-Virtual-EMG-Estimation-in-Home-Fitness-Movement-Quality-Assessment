import cv2
import os

def extract_video_segment(input_video_path, output_video_path, start_frame, end_frame):
    """
    Extract a given range of frames from a video and write it out as a new video.
    
    :param input_video_path: full path of the source video
    :param output_video_path: full path of the output video (.mp4 recommended)
    :param start_frame: start frame (integer)
    :param end_frame: end frame (integer)
    """
    # sanity-check the arguments
    if start_frame < 0 or end_frame <= start_frame:
        print("❌ Error: the start frame must be ≥ 0 and the end frame must be greater than the start frame!")
        return

    print(f"🎬 Reading source video: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("❌ Error: cannot open the source video; check that the path is correct.")
        return

    # get the original video's properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📊 Video info - FPS: {fps:.2f}, resolution: {width}x{height}, total frames: {total_frames}")

    # make sure the end frame does not exceed the video length
    if end_frame > total_frames:
        print(f"⚠️ Warning: the requested end frame ({end_frame}) exceeds the video length ({total_frames}); clamping to the last frame.")
        end_frame = total_frames

    # set the video codec (FourCC); mp4v is used here to write .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # create the VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 🔥 key: let OpenCV seek straight to the start frame, skipping the time spent reading earlier frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    print(f"✂️ Extracting frames {start_frame} to {end_frame}...")

    # read and write frame by frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Could not read a frame or reached the end of the video early.")
            break
        
        # write the frame into the new video
        out.write(frame)
        
        # show progress every 50 frames to avoid flooding the screen
        if current_frame % 50 == 0 or current_frame == end_frame:
            progress = ((current_frame - start_frame) / (end_frame - start_frame + 1e-5)) * 100
            print(f"⏳ progress: {current_frame}/{end_frame} frames ({progress:.1f}%)")

        current_frame += 1

    # release resources
    cap.release()
    out.release()
    print(f"✅ Cropping done! File saved to: {output_video_path}")

# ==========================================
# test and usage area
# ==========================================
if __name__ == "__main__":
    # replace with your actual paths and parameters
    SOURCE_VIDEO = r"<DATASET_ROOT>\0513\opoencap\S01\1-20\OpenCapData_c9bccd16-66a2-4039-a5f1-a133ff09e091\Videos\Cam0\InputMedia\general11-20\general11-20_sync.mp4"
    
    # .mp4 is recommended for the output file
    OUTPUT_VIDEO = r"<DATASET_ROOT>\Code\Test_video\S01_TEST_0513_Segment11-20.mp4"
    
    # set the frame range you want to extract (e.g. frames 150 to 400)
    START_FRAME = 210
    END_FRAME = 2223

    # run the function
    extract_video_segment(SOURCE_VIDEO, OUTPUT_VIDEO, START_FRAME, END_FRAME)