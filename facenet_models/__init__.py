from typing import NamedTuple, Optional

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import crop_resize

__all__ = ["FacenetModel"]


class _Detections(NamedTuple):
    boxes: np.ndarray
    probabilities: np.ndarray
    face_landmarks: np.ndarray


class FacenetModel:
    def __init__(self, device: str = "cpu"):
        """
        Parameters
        ----------
        device : str
            Indicates the device to place both models.
            E.g. 'cuda:0' will place the model on GPU 0.

            These models will automatically place images on the
            appropriate device during their forwards passes."""

        self._device = device.lower()
        self._mtcnn = MTCNN(device=self._device)
        self._resnet = InceptionResnetV1(
            pretrained="vggface2", device=self._device
        ).eval()

    def detect(self, image: np.ndarray) -> _Detections:
        """Detect faces in an image.

        Parameters
        ----------
        image : np.ndarray, shape=(H, W, 3)
            The image in which to detect faces.

        Returns
        -------
        Tuple[np.ndarray shape=(N, 4), np.ndarray shape=(N,), np.ndarray shape=(N, 5)]
            (boxes, probabilities, landmarks) where:
            - boxes is a shape-(N, 4) array of boxes, where N is the number of faces detected in the image.
              Each box is represented as (left, top, right, bottom).
            - probabilities is a shape-(N,) array of probabilities corresponding to each detected face.
            - landmarks is a shape-(N, 5) arrays of facial landmarks corresponding to each detected face.
        """
        boxes, probs, landmarks = self._mtcnn.detect(
            np.ascontiguousarray(image), landmarks=True
        )
        return _Detections(boxes=boxes, probabilities=probs, face_landmarks=landmarks)

    def compute_descriptors(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute descriptor vectors for the faces contained in ``boxes``.

        Parameters
        ----------
        image : np.ndarray, shape=(H, W, 3)
            The image in which to detect faces.

        boxes : np.ndarray, shape=(N, 4)
            The bounding boxes containing the faces for which to compute descriptors. The boxes should be
            given as (left, top, right, bottom).

        Returns
        -------
        np.ndarray, shape=(N, 512)
            The descriptor vectors, where N is the number of faces.
        """
        crops = np.array(
            [crop_resize(image, [int(max(0, coord)) for coord in box], 160)
             for box in boxes], dtype='int32'
        )
        crops = (torch.tensor(crops).float() - 127.5) / 128
        with torch.no_grad():
            return (
                self._resnet(crops.permute(0, 3, 1, 2).to(self._device)).cpu().numpy()
            )