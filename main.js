var socket = io("http://localhost:8000");
let captureButtonClicked = false;

socket.on("connect", function () {
  console.log("Connected...!", socket.connected);
});

async function playVideo() {
  const video = document.querySelector("#videoElement");
  const canvasOutput = document.querySelector("#canvasOutput");

  video.width = 500;
  video.height = 375;

  // Set canvas dimensions to match video dimensions
  canvasOutput.width = video.width;
  canvasOutput.height = video.height;

  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then(function (stream) {
        video.srcObject = stream;
        video.play();
      })
      .catch(function (err0r) {
        console.log(err0r);
        console.log("Something went wrong!");
      });
  } else {
    console.log("no video element?");
  }

  let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
  let cap = new cv.VideoCapture(video);

  const FPS = 22;

  setInterval(() => {
    if (captureButtonClicked) {
      cap.read(src);

      // Draw the video frame onto the canvas
      const ctx = canvasOutput.getContext("2d");
      ctx.drawImage(video, 0, 0, video.width, video.height);

      var type = "image/png";
      var data = canvasOutput.toDataURL(type);
      data = data.replace("data:" + type + ";base64,", ""); //split off junk at the beginning

      socket.emit("image", data);

      captureButtonClicked = false; // Reset the flag
    }
  }, 10000 / FPS);
}

async function onCVLoad() {
  if (cv.getBuildInformation) {
    console.log(cv.getBuildInformation());
    playVideo();
  } else {
    // WASM
    if (cv instanceof Promise) {
      cv = await cv;
      console.log(cv.getBuildInformation());
      playVideo();
    } else {
      cv["onRuntimeInitialized"] = () => {
        console.log(cv.getBuildInformation());
        playVideo();
      };
    }
  }
}
function captureImage() {
  captureButtonClicked = true;
}
socket.on("response_back", function (data) {
  const image_id = document.getElementById("image");
  console.log("response", data);

  // Display the image
  image_id.src = data.image;

  // Check if authorized and handle bounding box
  if (data.authorized) {
    // Handle authorized case
    console.log("Authorized person detected.");
  } else {
    // Handle unauthorized case
    console.log("Unauthorized person detected.");

    // Access bounding box information: data.boxes
    const canvasOutput = document.getElementById("canvasOutput");
    const ctx = canvasOutput.getContext("2d");

    // Draw bounding boxes on the canvas
    for (const box of data.boxes) {
      ctx.beginPath();
      ctx.rect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "red";
      ctx.fillStyle = "rgba(255, 0, 0, 0.3)";
      ctx.stroke();
      ctx.fill();
      ctx.closePath();
    }
  }
});
