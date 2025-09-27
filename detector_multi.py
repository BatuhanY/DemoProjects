import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model


class suppress_stderr:  # error stfu-inator, sadly cannot supress tensorflow
    def __enter__(self):
        if os.name == 'nt':
            self.null_f = 'NUL'
        else:
            self.null_f = '/dev/null'
        # Save the original stderr FD
        self.orig_fd = sys.stderr.fileno()
        self.saved_fd = os.dup(self.orig_fd)
        # Open null and redirect stderr to it
        self.devnull_fd = os.open(self.null_f, os.O_WRONLY)
        os.dup2(self.devnull_fd, self.orig_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stderr
        os.dup2(self.saved_fd, self.orig_fd)
        os.close(self.saved_fd)
        os.close(self.devnull_fd)

# globals
CLASS_NAMES = ['apple', 'none']
MIN_AREA = 500
MODEL_PATH     = "models/model_20250728_015433.keras"
WINDOW_SIZE    = (64, 64)       # must match the classifier input
PYR_SCALE      = 1.5            # image pyramid downscale factor
PYR_MIN_SIZE   = (64, 64)       # stop pyramid when smaller than this
WIN_STRIDE     = 32             # how many pixels to slide each step
CONF_THRESHOLD = 0.92           # only keep windows >= this apple-probability
NMS_IOU_THRESH = 0.3            # overlap suppression

def list_cameras(max_devices=5):
    available = []
    for i in range(max_devices):
        # Silence OpenCV errors while probing
        with suppress_stderr():
            cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue

        # Also silence any read errors
        with suppress_stderr():
            ret, _ = cap.read()

        if ret:
            available.append(i)
        cap.release()
    return available

def select_camera():
    cams = list_cameras()
    if not cams:
        print("No cameras detected.")
        return None
    print("Available camera devices:")
    for idx in cams:
        print(f"[{idx}] Camera {idx}")
    while True:
        choice = input("Enter camera index (or 'q' to quit): ")
        if choice.lower() == 'q':
            return None
        if choice.isdigit() and int(choice) in cams:
            return int(choice)
        print("Invalid selection. Try again.")

def open_camera(idx):
    with suppress_stderr():
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Failed to open camera {idx}.")
            return None
        print(f"Camera {idx} opened. Press 'c' to confirm, 'x' to cancel.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame not available, retrying...")
                continue
            cv2.imshow("Confirm Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                cv2.destroyWindow("Confirm Camera")
                return cap
            elif key == ord('x'):
                cap.release()
                cv2.destroyWindow("Confirm Camera")
                return None

def process_image_path():
    while True:
        print("\n-- Image File Mode --")
        print("You can enter either:")
        print(" 1. A relative path, e.g.  data/test/my_apple.jpg")
        print(" 2. An absolute path, e.g.  C:\\Users\\You\\Pics\\apple.png")
        print("If the path contains spaces, wrap it in quotes.")
        path = input("Enter image file path: ").strip().strip('"')

        if path.lower() in ('q', 'quit', ''):
            print("Operation cancelled by user.\n")
            return None  # User chose to exit, return nothing

        if not os.path.isfile(path):
            print(f"File not found at: {path}")
            print("Please check the path and try again.\n")
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load image. File may not be a valid image: {path}")
            print("Ensure the file is a supported image format and try again.\n")
            continue

        cv2.imshow("Input Image", img)
        print("Press any key to continue...")
        cv2.waitKey(5000)
        cv2.destroyWindow("Input Image")
        return img

def pyramid(image, scale=PYR_SCALE, minSize=PYR_MIN_SIZE):
# a gross simplification of how human vision cone works
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        if w < minSize[0] or h < minSize[1]:
            break
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        yield image

def sliding_window(image, step=WIN_STRIDE, windowSize=WINDOW_SIZE):
    for y in range(0, image.shape[0] - windowSize[1] + 1, step):
        for x in range(0, image.shape[1] - windowSize[0] + 1, step):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def non_max_suppression(dets, iou_thresh=NMS_IOU_THRESH):
    if not dets:
        return []
    arr = np.array(dets)
    x1, y1, x2, y2, scores = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(tuple(arr[i]))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= iou_thresh)[0] + 1]
    return keep

def detect_with_pyramid(image, model):
    detections = []
    for resized in pyramid(image):
        scale = image.shape[0] / float(resized.shape[0])
        for (x, y, window) in sliding_window(resized):
            # Prepare for model
            roi = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            arr = roi.astype('float32') / 255.0
            arr = np.expand_dims(arr, axis=(0, -1))  # shape (1,64,64,1)

            # Predict
            probs = model.predict(arr, verbose=0)[0]
            score = float(probs[0])  # apple class index is [0] btw
            if score >= CONF_THRESHOLD:
                x1 = int(x * scale)
                y1 = int(y * scale)
                x2 = int((x + WINDOW_SIZE[0]) * scale)
                y2 = int((y + WINDOW_SIZE[1]) * scale)
                detections.append((x1, y1, x2, y2, score))

    return non_max_suppression(detections)  # Suppress exact overlapping

def split_and_filter_apples(
    dets,
    image,
    heat_thresh=0.1,
    min_area_frac=0.001,
    sky_v_thresh=0.9,
    apple_hue_ranges=[(0,10),(160,180)],
    debug=False
    ):

    h, w = image.shape[:2]
    total_area = h*w
    min_area = total_area * min_area_frac

    # 1) Heatmap converted to mask
    heat = np.zeros((h, w), dtype=np.float32)
    for x1,y1,x2,y2,_ in dets:
        x1c,y1c = max(0,x1), max(0,y1)
        x2c,y2c = min(w,x2), min(h,y2)
        heat[y1c:y2c, x1c:x2c] += 1
    heat /= max(len(dets),1)
    bin_mask = (heat >= heat_thresh).astype(np.uint8)*255

    # 2) Distance transform + watershed
    # a) Light opening (only 1 iteration)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    opening = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # b) Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    # c) Distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # d) Sure foreground with lower threshold (0.2)
    _, sure_fg = cv2.threshold(dist, 0.2*dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    # e) Unknown handling
    unknown = cv2.subtract(sure_bg, sure_fg)
    # f) Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    # g) Watershed
    ws_img = image.copy()
    cv2.watershed(ws_img, markers)

    # 3) Collect boxes
    boxes = []
    for lbl in range(2, markers.max()+1):
        mask_lbl = (markers==lbl).astype(np.uint8)*255
        cnts,_ = cv2.findContours(mask_lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x,y,ww,hh = cv2.boundingRect(c)
        roi = image[y:y+hh, x:x+ww]

        # 4) Sky filter
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_mean = hsv[:,:,2].mean()/255.0
        if v_mean > sky_v_thresh:
            continue
        hue = hsv[:,:,0]
        mask_hue = np.zeros_like(hue, dtype=np.uint8)
        for low,high in apple_hue_ranges:
            mask_hue |= cv2.inRange(hue, low, high)
        if mask_hue.mean()/255.0 < 0.05:
            continue

        boxes.append((x,y,x+ww,y+hh))

    if debug:
        cv2.imshow("Mask", bin_mask)
        cv2.imshow("Opening", opening)
        cv2.imshow("Dist", (dist/dist.max()*255).astype(np.uint8))
        cv2.imshow("Sure FG", sure_fg)
        cv2.imshow(
            "Markers",
            (markers.astype(np.float32)/markers.max()*255).astype(np.uint8)
            )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return boxes

    if debug:
        # visualize intermediate masks
        cv2.imshow("Heatmap mask", bin_mask)
        cv2.imshow("Distance transform", (dist/dist.max()*255).astype(np.uint8))
        cv2.imshow(
            "Watershed markers",
            (markers.astype(np.float32)/markers.max()*255).astype(np.uint8)
            )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return boxes

def resizinator(image, target_size=(640, 480)):
    # otherwise the pyramid detection will take ages
    # Resize `image` so that it best fits within `target_size`, preserving aspect ratio.
    # If the image is larger than the target in at least one dimension,
    # it will be shrunk.
    # If the image is smaller than the target in both dimensions, it will be enlarged.
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Compute ratios of target to original
    ratio_w = target_w / w
    ratio_h = target_h / h

    # Decide whether to shrink or enlarge:
    # If either dimension is too big (ratio < 1), we shrink by the smaller ratio.
    # Otherwise both dims are smaller than target (ratio ≥ 1)
    # so we enlarge by the larger ratio.
    if ratio_w < 1 or ratio_h < 1:
        scale = min(ratio_w, ratio_h)   # shrink so both dims =< target
    else:
        scale = max(ratio_w, ratio_h)   # enlarge so at least one dim == target

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        )
    return resized

def main():
    model = load_model(MODEL_PATH)
    shut_it_down = False        # just a flag
    while not shut_it_down:     # user interface loop
        print("Select input mode:")
        print("[1] Image file")
        print("[2] Camera feed")
        choice = input("Enter choice (1 or 2, 'q' to quit): ")

        if choice.lower() == 'q':
            print("Operation cancelled. Exiting all processes.")
            shut_it_down = True
            break

        if choice == '1':
            while True:
                captured_image = process_image_path()
                if captured_image is not None:
                    print("Image captured.")
                    break
                else:
                    print("Process failed. Restarting image input.")
            break

        elif choice == '2':
            cam_idx = select_camera()
            if cam_idx is None:
                return
            cap = open_camera(cam_idx)
            if cap is None:
                print("Camera not confirmed. Exiting.")
                return
            print("Camera confirmed. Starting feed. Press 'q' to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                cv2.imshow(
                    "Camera is live - Press SPACE to capture, 'q' to quit",
                    frame
                    )
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Exiting camera feed.")
                    break
                elif key == 32:  # SPACE key
                    captured_image = frame.copy()
                    print("Image captured.")
                    break  # for now allow only one image to be processed
            cap.release()
            cv2.destroyAllWindows()
            break
        else:
            print("Bad key. Try again.")
    if not shut_it_down:
    # after capturing or loading an image into captured_image, run detection
        captured_image = resizinator(captured_image)
        dets = detect_with_pyramid(captured_image, model)
        dets = [
            tuple(int(v) if i < 4 else float(v) for i, v in enumerate(item))
            for item in dets
            ]
        print(f"Detections: {len(dets)} windows")

        boxes = split_and_filter_apples(
            dets,
            captured_image,
            heat_thresh=0.05,
            min_area_frac=0.002,
            sky_v_thresh=0.9,
            apple_hue_ranges=[(0,10),(160,180)],
            debug=True
            )

    for (x1,y1,x2,y2) in boxes:     # mark boundding boxes on display
        cv2.rectangle(captured_image, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("Adaptive Multi-Apple Detections", captured_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
