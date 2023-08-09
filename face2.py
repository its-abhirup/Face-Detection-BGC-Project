import cv2
import face_recognition
import os

# Path to the directory containing the known face images
known_faces_dir = r"C:\Users\abhir\OneDrive\Documents\Programs\Project\known_faces"

# Initialize arrays for known face encodings and names
known_face_encodings = []
known_face_names = []

# Load the known face images and encode them
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        face_image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        face_encoding = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to RGB for face recognition
    rgb_frame = frame[:, :, ::-1]

    # Find faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Compare each face in the current frame to the known faces
    for face_encoding in face_encodings:
        # Compare the current face encoding with the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Check if there is a match
        if True in matches:
            # Find the indices of all matched faces
            match_indices = [i for i, match in enumerate(matches) if match]
            match_names = [known_face_names[i] for i in match_indices]

            # Use the first matched face name
            name = match_names[0]

        # Draw a rectangle and label around the face
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, name, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
