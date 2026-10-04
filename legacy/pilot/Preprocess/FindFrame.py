import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_video_range(cap, start_frame, end_frame, roi):
    """Helper that counts red pixels within a given frame interval"""
    x, y, w, h = roi
    red_pixel_counts = []
    frame_indices = []

    # relaxed thresholds: also detect whitish or dark reds
    lower_red1 = np.array([0, 70, 80])  
    upper_red1 = np.array([5, 170, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([179, 255, 255])

    # seek to the interval's start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        roi_frame = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        full_mask = cv2.bitwise_or(mask1, mask2)
        
        count = cv2.countNonZero(full_mask)
        red_pixel_counts.append(count)
        frame_indices.append(current_frame)
        
        current_frame += 1

    return np.array(frame_indices), np.array(red_pixel_counts)


def find_sync_intervals(video_path, start_window, end_window):
    # 1. read the video and basic info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: cannot open the video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: FPS = {fps:.2f}, total frames = {total_frames}")

    # compute the start and end frames of the two target intervals
    start_min_f = int(start_window[0] * fps)
    start_max_f = min(int(start_window[1] * fps), total_frames)
    end_min_f = int(end_window[0] * fps)
    end_max_f = min(int(end_window[1] * fps), total_frames)

    print(f"🔎 searching the [start, light on] interval: frame {start_min_f} to frame {start_max_f}")
    print(f"🔎 searching the [end, light off] interval: frame {end_min_f} to frame {end_max_f}")

    # ==========================================
    # 2. select an ROI for each interval
    # ==========================================
    # --- (A) select the start ROI ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_min_f)
    ret, frame_start = cap.read()
    if not ret:
        print("Error: cannot read the start interval's first frame")
        return

    print("\n(1/2) In the window, select the indicator-light area for the [start interval]...")
    roi_start = cv2.selectROI("Select START LED ROI", frame_start, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select START LED ROI")
    if roi_start == (0, 0, 0, 0):
        print("No start region selected; exiting.")
        return

    # --- (B) select the end ROI ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, end_min_f)
    ret, frame_end = cap.read()
    if not ret:
        print("Error: cannot read the end interval's first frame")
        return

    print("\n(2/2) In the window, select the indicator-light area for the [end interval]...")
    roi_end = cv2.selectROI("Select END LED ROI", frame_end, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select END LED ROI")
    if roi_end == (0, 0, 0, 0):
        print("No end region selected; exiting.")
        return

    # ==========================================
    # 3. analyse the two intervals
    # ==========================================
    print("\nAnalysing pixel changes in the [start interval]...")
    frames_start, counts_start = analyze_video_range(cap, start_min_f, start_max_f, roi_start)
    
    print("Analysing pixel changes in the [end interval]...")
    frames_end, counts_end = analyze_video_range(cap, end_min_f, end_max_f, roi_end)
    
    cap.release()

    # ==========================================
    # 4. detect the change points
    # ==========================================
    # 🌟 light on (start interval): largest difference (most positive)
    diffs_start = np.diff(counts_start)
    jump_on_idx = np.argmax(diffs_start)
    absolute_sync_frame_on = frames_start[jump_on_idx + 1]

    # 🌟 light off (end interval): smallest difference (most negative)
    diffs_end = np.diff(counts_end)
    jump_off_idx = np.argmin(diffs_end)
    absolute_sync_frame_off = frames_end[jump_off_idx + 1]
    
    # 🌟 compute 180 frames before the light-off (never below 0)
    frame_minus_180 = max(0, absolute_sync_frame_off - 180)

    print("\n" + "=" * 55)
    print(f"✅ Sync points and target frames found!")
    print(f"🟢 [recording start] light on: frame {absolute_sync_frame_on} ({absolute_sync_frame_on/fps:.3f} s)")
    print(f"🔴 [recording end] light off: frame {absolute_sync_frame_off} ({absolute_sync_frame_off/fps:.3f} s)")
    print(f"🟡 [back-calculated] 180 frames before light-off: frame {frame_minus_180} ({frame_minus_180/fps:.3f} s)")
    print("=" * 55 + "\n")

    # ==========================================
    # 5. extract and plot neighbouring frames (three rows)
    # ==========================================
    print("Extracting neighbouring frames for confirmation...")
    cap = cv2.VideoCapture(video_path) 
    
    look_back = 5
    look_forward = 5
    
    fig_frames, axes = plt.subplots(3, look_back + look_forward + 1, figsize=(18, 9))
    fig_frames.suptitle("Verification: ON (Top), OFF (Middle), and OFF - 180 Frames (Bottom)", fontsize=16, fontweight='bold')

    # 🌟 key change: map each event to its own ROI
    sync_points = [
        (absolute_sync_frame_on, "ON", "green", roi_start),
        (absolute_sync_frame_off, "OFF", "red", roi_end),
        (frame_minus_180, "-180F", "orange", roi_end) # -180F reuses the end ROI
    ]

    for row_idx, (sync_f, label, color, current_roi) in enumerate(sync_points):
        x, y, w, h = current_roi
        extract_start = max(0, sync_f - look_back)
        cap.set(cv2.CAP_PROP_POS_FRAMES, extract_start)
        
        for i in range(look_back + look_forward + 1):
            curr_frame_num = extract_start + i
            ret, frame = cap.read()
            if not ret:
                break
            
            # crop with each event's own ROI
            roi_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            ax = axes[row_idx, i]
            ax.imshow(roi_rgb)
            ax.axis('off')
            
            if curr_frame_num == sync_f:
                ax.set_title(f"[{curr_frame_num}]\nTarget {label}", color=color, fontweight='bold')
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(4)
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_title(f"{curr_frame_num}")

    cap.release()

    # ==========================================
    # 6. plot line charts for both intervals (side by side)
    # ==========================================
    fig_plot, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig_plot.suptitle("LED Brightness over Time (Split Windows)", fontsize=14)

    # left: start (light on) interval
    ax1.plot(frames_start, counts_start, color='blue', marker='o', markersize=3, label='Red Pixels')
    ax1.axvline(x=absolute_sync_frame_on, color='green', linestyle='--', linewidth=2, label=f'Sync ON ({absolute_sync_frame_on})')
    ax1.set_title("Start Window (Finding ON)")
    ax1.set_xlabel("Absolute Frame Number")
    ax1.set_ylabel("Number of Red Pixels")
    ax1.legend()
    ax1.grid(True)

    # right: end (light off) interval
    ax2.plot(frames_end, counts_end, color='blue', marker='o', markersize=3, label='Red Pixels')
    ax2.axvline(x=absolute_sync_frame_off, color='red', linestyle='--', linewidth=2, label=f'Sync OFF ({absolute_sync_frame_off})')
    
    # if frame -180 falls within the right chart's interval, draw an orange dashed line there too
    if end_min_f <= frame_minus_180 <= end_max_f:
        ax2.axvline(x=frame_minus_180, color='orange', linestyle='--', linewidth=2, label=f'-180F ({frame_minus_180})')
    
    ax2.set_title("End Window (Finding OFF)")
    ax2.set_xlabel("Absolute Frame Number")
    ax2.set_ylabel("Number of Red Pixels")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# ==========================================
# execution area
# ==========================================
if __name__ == "__main__":
    VIDEO_FILE = r"<DATASET_ROOT>\0625\Open\OpenCapData_S01\Videos\Cam0\InputMedia\C\C_sync.mp4"
    START_WINDOW = (1, 2)   # interval in which to look for the light turning on
    END_WINDOW = (31, 32)    # interval in which to look for the light turning off
    
    find_sync_intervals(VIDEO_FILE, START_WINDOW, END_WINDOW)