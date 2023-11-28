from face_rec import show_image_with_recognition
from face_rec.whispers import labeled_pics_to_descriptors
from pathlib import Path
from scipy.spatial import distance

def is_authorized(img):
    _, _, matches, boxes, descriptors = show_image_with_recognition(img)
    authorized_descriptors, authorized_names, _ = labeled_pics_to_descriptors(Path("face_rec") / "pics")
    
    authorized = False
    for box, descriptor in zip(boxes, descriptors):
        for auth_descriptor, auth_name in zip(authorized_descriptors, authorized_names):
            if distance.euclidean(descriptor, auth_descriptor) < 0.1:  # 0.1 is a threshold, can be adjusted
                print(f"{auth_name} is authorized.")
                authorized = True
                break
    
    return authorized, boxes
