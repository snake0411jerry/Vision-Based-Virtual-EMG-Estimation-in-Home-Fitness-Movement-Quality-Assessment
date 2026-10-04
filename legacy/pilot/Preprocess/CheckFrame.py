import cv2
import numpy as np

# ==========================================
# 🔴 set the HSV threshold you want to test here
# ==========================================
LOWER_RED1 = np.array([0, 70, 80])  
UPPER_RED1 = np.array([5, 170, 255])
LOWER_RED2 = np.array([160, 60, 60])
UPPER_RED2 = np.array([179, 255, 255])


# ==========================================
# mouse click callback (query HSV)
# ==========================================
def mouse_callback(event, x, y, flags, param):
    hsv_frame, scale, original_w = param

    if event == cv2.EVENT_LBUTTONDOWN:
        # decide whether the click is on the left half (original image) or the right half (mask image)
        magnified_w = original_w * scale
        
        if x < magnified_w:
            # clicked the left half
            original_roi_x = int(x / scale)
            clicked_side = "left (original image)"
        else:
            # clicked the right half
            original_roi_x = int((x - magnified_w) / scale)
            clicked_side = "right (mask image)"

        original_roi_y = int(y / scale)

        # make sure the coordinates are within range
        if 0 <= original_roi_x < hsv_frame.shape[1] and 0 <= original_roi_y < hsv_frame.shape[0]:
            h, s, v = hsv_frame[original_roi_y, original_roi_x]
            print("-" * 30)
            print(f"👉 clicked area: {clicked_side}")
            print(f"✅ exact HSV value: [H:{h}, S:{s}, V:{v}]")
            print("-" * 30)

# ==========================================
# main program
# ==========================================
def test_hsv_thresholds(video_path, frame_index, magnification=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: cannot read that frame.")
        return

    print(f"\n[Step 1] Showing frame {frame_index} of the video.")
    print("Select the region containing the LED, then press 'SPACE' or 'ENTER'.")
    
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    if roi == (0, 0, 0, 0):
        print("Nothing selected; exiting.")
        return

    x_start, y_start, w, h = roi
    roi_frame = frame[y_start:y_start+h, x_start:x_start+w]
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # 1. build the mask
    mask1 = cv2.inRange(hsv_roi, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv_roi, LOWER_RED2, UPPER_RED2)
    full_mask = cv2.bitwise_or(mask1, mask2)

    # 2. count red pixels
    red_pixel_count = cv2.countNonZero(full_mask)
    print(f"\n====================================")
    print(f"🔥 total red pixels found in this ROI: {red_pixel_count} pixels")
    print(f"====================================\n")

    # 3. enlarge the image (INTER_NEAREST keeps the pixels blocky)
    new_w, new_h = w * magnification, h * magnification
    mag_roi = cv2.resize(roi_frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mag_mask = cv2.resize(full_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # convert the single-channel mask to three-channel BGR so it can sit beside the original
    mag_mask_bgr = cv2.cvtColor(mag_mask, cv2.COLOR_GRAY2BGR)

    # 4. add text labels
    cv2.putText(mag_roi, "Original ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(mag_mask_bgr, f"Red Mask: {red_pixel_count} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 5. show side by side (original on the left, mask on the right)
    combined_view = np.hstack((mag_roi, mag_mask_bgr))

    window_name = "HSV Threshold Debugger (Click to see HSV)"
    cv2.namedWindow(window_name)
    
    # bind mouse events
    cv2.setMouseCallback(window_name, mouse_callback, (hsv_roi, magnification, w))

    print("[Step 2] Comparison image shown.")
    print(" - left: original, enlarged")
    print(" - right: red pixels meeting the threshold (shown bright)")
    print("👉 You can click “red pixels that were not captured” in the left image to see their HSV values in the terminal.")
    print("👉 Press any key to close the window.")

    cv2.imshow(window_name, combined_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================
# execution area
# ==========================================
if __name__ == "__main__":
    # replace with your video path
    VIDEO_PATH = r"<DATASET_ROOT>\0529\opencap\OpenCapData_4a5f4e24-561c-4552-8423-5593a5306f18_66922d47-891b-47d7-9258-591472bc36e8\OpenCapData_4a5f4e24-561c-4552-8423-5593a5306f18\Videos\Cam0\InputMedia\compen-b1-10\compen-b1-10_sync.mp4"
    # ⚠️ set your "two" time intervals (in seconds)
    # enter the frame number you want to test (e.g. the frame in your chart where the light turns on, or the frame just before it turns off)
    FRAME_TO_INSPECT = 341

    # magnification (default 15×, so the cells are large and easy to click)
    MAGNIFICATION = 15

    test_hsv_thresholds(VIDEO_PATH, FRAME_TO_INSPECT, MAGNIFICATION)