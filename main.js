var socket = io('http://localhost:8000');

socket.on('connect', function () {
    console.log("Connected...!", socket.connected)
});

async function playVideo() {

    const video = document.querySelector("#videoElement");
    
    video.width = 500;
    video.height = 375;;
    
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            video.srcObject = stream;
            video.play();
        })
        .catch(function (err0r) {
            console.log(err0r)
            console.log("Something went wrong!");
        });
    } else {
        console.log("no video element?")
    }
    
    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let cap = new cv.VideoCapture(video);
    
    const FPS = 22;
    
    setInterval(() => {
        cap.read(src);
        
        var type = "image/png"
        var data = document.getElementById("canvasOutput").toDataURL(type);
        data = data.replace('data:' + type + ';base64,', ''); //split off junk at the beginning
        
        socket.emit('image', data);
    }, 10000 / FPS);   
}

async function onCVLoad() {
    if (cv.getBuildInformation) {
        console.log(cv.getBuildInformation());
        playVideo();
    }
    else {
        // WASM
        if (cv instanceof Promise) {
            cv = await cv;
            console.log(cv.getBuildInformation());
            playVideo();
        } else {
            cv['onRuntimeInitialized'] = () => {
                console.log(cv.getBuildInformation());
                playVideo();
            }
        }
    }
}

socket.on('response_back', function (image) {
    const image_id = document.getElementById('image');
    console.log("response", image);
    image_id.src = image;
});