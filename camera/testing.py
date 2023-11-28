#Testing if camera ports are funky
import cv2

# Try to open the default camera (index 0)
cap = cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open the default camera.")

    # If the default camera is not available, try to open any available camera
    for index in range(10):  # Try the first 10 camera indices
        cap = cv2.VideoCapture(index)

        if cap.isOpened():
            print(f"Successfully opened camera with index {index}.")
            break

if cap.isOpened():
    # Capture a frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if ret:
        # Display the captured frame
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(0)
    else:
        print("Error: Failed to capture a frame from the camera.")

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()
