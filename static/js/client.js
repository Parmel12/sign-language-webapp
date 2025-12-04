const socket = io();
const video = document.getElementById("webcam");
const signLabel = document.getElementById("sign");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

function captureFrame() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataURL = canvas.toDataURL("image/jpeg");
    socket.emit("frame", dataURL);
}

// Receive detected sign from server
socket.on("sign_update", data => {
    signLabel.innerText = data;
});

// Send frame every 0.5 seconds
setInterval(captureFrame, 500);
