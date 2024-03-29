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
      canvasOutput.classList.remove("hidden");
      console.log(canvasOutput.classList)
      cap.read(src);

      // Draw the video frame onto the canvas
      const ctx = canvasOutput.getContext("2d");
      ctx.drawImage(video, 0, 0, video.width, video.height);

      var type = "image/png";
      var image = canvasOutput.toDataURL(type);
      image = image.replace("data:" + type + ";base64,", ""); //split off junk at the beginning

      const name = document.getElementById('new-user-name').value;
      const data = {image, name};
      console.log("value of name: ", data)
      
      socket.emit("new-auth-image", data);

      captureButtonClicked = false; // Reset the flag
    }
  }, 10_000 / FPS);
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

socket.on("response_new_auth", function (data) {
  canvasOutput = document.getElementById("canvasOutput")
  canvasOutput.classList.add("hidden");
  console.log(canvasOutput.classList)

  const image_id = document.getElementById("image");
  const authorizationStatus = document.getElementById("authorizationStatus");

  // Display the image
  image_id.src = data.image;

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

  // Check if authorized and handle bounding box
  if (!data.authorized && data.boxes) {
    const canvasOutput = document.getElementById("canvasOutput");
    const ctx = canvasOutput.getContext("2d");
  }
});
