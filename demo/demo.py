import cv2
import torch
import numpy as np
from PIL import Image
import clip

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define scoring-related text prompts
scoring_prompts = [
    "a basketball going through the hoop",
    "a player shooting the ball",
    "a basketball dunk",
    "a basketball score"
]

# Preprocess text prompts
text_inputs = torch.cat([clip.tokenize(prompt) for prompt in scoring_prompts]).to(device)

# Function to analyze a frame for scoring events
def analyze_frame(frame):
    # Preprocess the frame for CLIP
    image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)

    # Get image and text features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    # Compute similarity between image and text prompts
    logits_per_image, _ = model(image, text_inputs)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Get the most likely scoring event
    scoring_event_index = np.argmax(probs)
    scoring_event_prob = probs[0, scoring_event_index]

    return scoring_event_index, scoring_event_prob

# Load video
video_path = "basketball_game.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize score counter
score = 0

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 360))

    # Analyze frame for scoring events
    scoring_event_index, scoring_event_prob = analyze_frame(frame)

    # If a scoring event is detected with high confidence, increment the score
    if scoring_event_prob > 0.8:  # Adjust threshold as needed
        score += 1
        print(f"Score detected! Total score: {score}")

    # Display the frame
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Basketball Game", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
