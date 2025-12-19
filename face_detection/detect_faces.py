import cv2
import torch
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt

# Initialize MTCNN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

def detect_faces(image_path):
    # Read image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, probs = mtcnn.detect(img_rgb)

    if boxes is None:
        print("No faces detected")
        return

    # Draw bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show result
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    detect_faces("data/event_photos/test.jpg")
