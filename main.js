var socket = io("http://localhost:8000");
let captureButtonClicked = false;

socket.on("connect", function () {
  console.log("Connected...!", socket.connected);
});

async function playVideo() {
  const video = document.querySelector("#videoElement");
  const canvasOutput = document.querySelector("#canvasOutput");
  const image = document.querySelector("#output-image");

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
      video.classList.add('hidden');
      image.classList.add('hidden');
      canvasOutput.classList.remove('hidden')

      // Draw the video frame onto the canvas
      const ctx = canvasOutput.getContext("2d");
      ctx.drawImage(video, 0, 0, video.width, video.height);

      var type = "image/png";
      var data = canvasOutput.toDataURL(type);
      data = data.replace("data:" + type + ";base64,", ""); //split off junk at the beginning

      socket.emit("image", data);

      captureButtonClicked = false; // Reset the flag
    }
  }, 1000 / FPS);
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
  const video = document.querySelector("#videoElement");
  const canvasOutput = document.querySelector("#canvasOutput");
  const image = document.querySelector("#output-image");

  const authorizationStatus = document.getElementById("authorizationStatus");

  video.classList.add('hidden');
  canvasOutput.classList.add('hidden');
  image.classList.remove('hidden');

  if (data.error) {
    // Handle error case
    console.log("No Face Detected");
    authorizationStatus.textContent = "No Face Detected.";
    authorizationStatus.style.color = "black";

    image.src = ""
  } else {
    // Display the image
    image.src = data.image;
    
    // Update the authorization status
    if (data.authorized) {
      // Handle authorized case
      console.log("Authorized person detected.");
      authorizationStatus.textContent = "Authorized person detected.";
      authorizationStatus.style.color = "green";
    } else {
      // Handle unauthorized case
      console.log("Unauthorized person detected.");
      authorizationStatus.textContent = "Unauthorized person detected.";
      authorizationStatus.style.color = "red";
    }
  }
    
  // Check if authorized and handle bounding box
  if (!data.authorized && data.boxes) {
    const canvasOutput = document.getElementById("canvasOutput");
    const ctx = canvasOutput.getContext("2d");
  }

  setTimeout(() => {
    video.classList.remove('hidden');
    canvasOutput.classList.add('hidden');
    image.classList.add('hidden');
    authorizationStatus.textContent = "Push Unlock Button to Access";
    authorizationStatus.style.color = "black";
  }, 5000);
});
