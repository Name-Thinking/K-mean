# pyrefly: ignore [missing-import]
import cv2
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
name = 'parinya'
os.makedirs(name, exist_ok=True)

# Find next image index to avoid overwriting
existing = [int(f.split('.')[0]) for f in os.listdir(name) if f.endswith('.jpg') and f.split('.')[0].isdigit()]
i = max(existing) + 1 if existing else 1

print(f"Saving to '{name}' folder. Starting at index {i}.")
print("Controls: Press 's' to save face photo, 'q' or 'ESC' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from camera or camera disconnected.")
        break

    cv2.rectangle(frame, (250, 120), (390, 300), (0, 0, 255), 2)
    face = cv2.cvtColor(frame[120:300, 250:390, :], cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('face', face)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        save_path = f'{name}/{i}.jpg'
        cv2.imwrite(save_path, face)
        print(f"Saved: {save_path}")
        i += 1
    elif key == ord('q') or key == 27:
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

