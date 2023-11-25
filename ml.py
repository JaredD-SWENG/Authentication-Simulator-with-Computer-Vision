# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score

# digits = load_digits()
    
# # Randomly split data into 70% training and 30% test
# X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=.30)
    
# clf_KNN = KNeighborsClassifier()


# def train():

#     # Train a kNN model using the training set
#     clf_KNN.fit(X_train, y_train)
	

# def prediction():
            
#     # Predictions using the kNN model on the test set
#     print("Predicting labels of the test data set - %i random samples" % (len(X_test)))
#     result = clf_KNN.predict(X_test)
    
#     accuracy = accuracy_score(y_test, result)
#     precision = precision_score(y_test, result, average="macro")
#     recall = recall_score(y_test, result, average="macro")
#     return [accuracy, precision, recall]

from face_rec import show_image_with_recognition
from face_rec.whispers import labeled_pics_to_descriptors
from pathlib import Path
from scipy.spatial import distance

def is_authorized(img):
    _, _, matches, _, descriptors = show_image_with_recognition(img)
    authorized_descriptors, authorized_names, _ = labeled_pics_to_descriptors(Path("face_rec") / "pics")
    
    for descriptor in descriptors:
        for auth_descriptor, auth_name in zip(authorized_descriptors, authorized_names):
            if distance.euclidean(descriptor, auth_descriptor) < 0.6:  # 0.6 is a threshold, can be adjusted

                print(f"{auth_name} is authorized.")
                return True
    print("Person is not authorized.")
    return False