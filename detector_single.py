import cv2
import os, sys
import numpy as np
from tensorflow.keras.models import load_model

class suppress_stderr:	# error stfu-inator, sadly cannot supress tensorflow
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
WINDOW_SIZE    = (64, 64)       # must match your classifier input
PYR_SCALE      = 1.5            # image pyramid downscale factor
PYR_MIN_SIZE   = (64, 64)       # stop pyramid when smaller than this
WIN_STRIDE     = 32             # how many pixels to slide each step
CONF_THRESHOLD = 0.9            # only keep windows ≥ this Apple-probability
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

def pyramid(image, scale=PYR_SCALE, minSize=PYR_MIN_SIZE):	# a gross simplification of how human vision cone works
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
	if not dets: return []
	arr = np.array(dets)
	x1, y1, x2, y2, scores = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
	areas = (x2 - x1) * (y2 - y1)
	order = scores.argsort()[::-1]
	keep = []
	while order.size:
		i = order[0]; keep.append(tuple(arr[i]))
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
			score = float(probs[0])  # apple class index is [0]
			if score >= CONF_THRESHOLD:
				x1 = int(x * scale)
				y1 = int(y * scale)
				x2 = int((x + WINDOW_SIZE[0]) * scale)
				y2 = int((y + WINDOW_SIZE[1]) * scale)
				detections.append((x1, y1, x2, y2, score))

	return non_max_suppression(detections)	# Suppress exact overlapping

def merge_detections_by_heatmap(dets, image_shape, thresh=0.5):		# Merge overlapping detections into one box via heatmap thresholding.
	if not dets:
		return None

	h, w = image_shape[:2]
	heat = np.zeros((h, w), dtype=np.float32)
	
	# 1) Accumulate
	for (x1, y1, x2, y2, _) in dets:
		# Clip to image bounds just in case
		x1c, y1c = max(0, x1), max(0, y1)
		x2c, y2c = min(w, x2), min(h, y2)
		heat[y1c:y2c, x1c:x2c] += 1.0

	# 2) Normalize
	heat /= len(dets)

	# 3) Threshold to binary mask
	mask = (heat >= thresh).astype(np.uint8) * 255

	# 4) Find the largest connected component in mask
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None

	# Pick the contour with the largest area
	best = max(contours, key=cv2.contourArea)
	ux, uy, uw, uh = cv2.boundingRect(best)
	return (ux, uy, ux + uw, uy + uh)

def main():
	model = load_model(MODEL_PATH)
	shut_it_down = False		# just a flag
	while not shut_it_down:		# user interface loop. no actual work here, just to grab an image
		print("Select input mode:")
		print("[1] Image file")
		print("[2] Camera feed")
		choice = input("Enter choice (1 or 2, 'q' to quit): ")

		if choice.lower() == 'q':
			print("Operation cancelled. Exiting all processes.")
			shut_it_down = True		# comment redacted
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
				cv2.imshow("Camera is live - Press SPACE to capture, 'q' to quit", frame)
				key = cv2.waitKey(1) & 0xFF

				if key == ord('q'):
					print("Exiting camera feed.")
					break
				elif key == 32:  # SPACE key
					captured_image = frame.copy()
					print("Image captured.")
					# optional: save to disk. placeholder, does not have dynamic file naming right now
					# cv2.imwrite("captured_image.png", captured_image)
					# print("Image saved to captured_image.png")
					break  # for now allow only one image to be processed
			cap.release()
			cv2.destroyAllWindows()
			break
		else:
			print("Bad key. Try again.")
	if not shut_it_down:		# after capturing or loading an image into captured_image, run detection
		dets = detect_with_pyramid(captured_image, model)
		dets = [tuple(int(v) if i < 4 else float(v) for i, v in enumerate(item)) for item in dets]
		print(f"Detections: {len(dets)} windows")

		box = merge_detections_by_heatmap(dets, captured_image.shape, thresh=0.1)
		if box:
			x1, y1, x2, y2 = box
			cv2.rectangle(captured_image, (x1, y1), (x2, y2), (0,255,0), 2)
			cv2.putText(captured_image, "APPLE", (x1, y1-5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
			cv2.imshow("Merged Detection", captured_image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

if __name__ == "__main__":
	main()