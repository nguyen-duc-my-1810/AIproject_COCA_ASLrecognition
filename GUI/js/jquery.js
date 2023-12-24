let socket;
let webcamActive = false;
let videoActive = false;

function activateWebcam() {
  socket = new WebSocket("ws://127.0.0.1:8000/ws");
  const videoCanvas = document.getElementById("video-canvas");
  const ctx = videoCanvas.getContext("2d");

  socket.onopen = () => {
    console.log("WebSocket connected");
    const request = { command: "activate_webcam" };
    socket.send(JSON.stringify(request));
    webcamActive = true;
  };

  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
  };

  socket.onclose = () => {
    console.log("WebSocket closed");
  };

  socket.onmessage = function (event) {
    if (webcamActive) {
      const frameBase64 = event.data;
      const blob = new Blob([frameBase64], { type: "image/jpeg" });
      const imageUrl = URL.createObjectURL(blob);

      const img = new Image();
      img.onload = function () {
        ctx.drawImage(img, 0, 0, videoCanvas.width, videoCanvas.height);
      };
      img.src = imageUrl;
      const data = JSON.parse(event.data);
      updateUI(data);
    }
  };
}

function deactivateWebcam() {
  webcamActive = false;
  const request = { command: "deactivate_webcam" };
  socket.send(JSON.stringify(request));

  const videoCanvas = document.getElementById("video-canvas");
  const ctx = videoCanvas.getContext("2d");
  ctx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
}

const cameraButton = document.getElementById("cameraButton");
cameraButton.addEventListener("click", () => {
  if (!webcamActive) {
    activateWebcam();
  } else {
    deactivateWebcam();
  }
});

async function changeOptions() {
  const modelOptions = document.getElementsByName("model");
  let selectedModel;
  for (const option of modelOptions) {
    if (option.checked) {
      selectedModel = option.value;
      break;
    }
  }

  const thresholdOptions = document.getElementsByName("threshold");
  let selectedThreshold;
  for (const option of thresholdOptions) {
    if (option.checked) {
      selectedThreshold = option.value;
      break;
    }
  }

  const modelWebsocket = new WebSocket("ws://127.0.0.1:8000/model");
  modelWebsocket.onopen = () => {
    const modelData = { selected_model: selectedModel };
    modelWebsocket.send(JSON.stringify(modelData));
  };

  const thresholdWebsocket = new WebSocket("ws://127.0.0.1:8000/threshold");
  thresholdWebsocket.onopen = () => {
    const thresholdData = { selected_hand: selectedThreshold };
    thresholdWebsocket.send(JSON.stringify(thresholdData));
  };
}

function updateUI(data) {
  const accumulatedWord = document.getElementById("accumulatedWord");
  accumulatedWord.innerText = data.accumulated_word;

  const sentenceWord1 = document.getElementById("sentence_word1");
  sentenceWord1.innerText = "Suggested sentence 1: \n" + data.sentence1;

  const sentenceWord2 = document.getElementById("sentence_word2");
  sentenceWord2.innerText = "Suggested sentence 2: \n" + data.sentence2;
}

function readText(text) {
  const speech = new SpeechSynthesisUtterance();
  speech.text = text;

  const voiceSelect = document.getElementsByName("voiceSelect");
  let selectedVoiceName;
  for (const voice of voiceSelect) {
    if (voice.checked) {
      selectedVoiceName = voice.value;
      break;
    }
  }

  const voices = speechSynthesis.getVoices();
  const desiredVoice = voices.find((voice) => voice.name === selectedVoiceName);
  if (desiredVoice) {
    speech.voice = desiredVoice;
    speechSynthesis.speak(speech);
  }
}

const readButton = document.getElementById("readButton");
readButton.addEventListener("click", function () {
  const accumulatedWord = document.getElementById("accumulatedWord");
  console.log(accumulatedWord);
  const textToRead = accumulatedWord.textContent;
  console.log(textToRead);
  readText(textToRead);
});

function clearAccumulatedWord() {
  const request = { command: "clear_accumulated_word" };
  socket.send(JSON.stringify(request));
  const accumulatedWord = document.getElementById("accumulatedWord");
  accumulatedWord.innerText = "";

  const sentenceWord1 = document.getElementById("sentence_word1");
  sentenceWord1.innerText = "Suggested sentence 1: ";

  const sentenceWord2 = document.getElementById("sentence_word2");
  sentenceWord2.innerText = "Suggested sentence 2: ";
}

function activatevideo() {
  socket = new WebSocket("ws://127.0.0.1:8000/ws_video");
  const videoCanvas = document.getElementById("video-canvas");
  const ctx = videoCanvas.getContext("2d");
  const input = document.getElementById("fileInput");
  const fileName = input.files[0].name;

  socket.onopen = () => {
    console.log("WebSocket connected");

    const request = { command: "activate_video" };
    socket.send(JSON.stringify(request));

    const fileData = { file_name: fileName };
    socket.send(JSON.stringify(fileData));
    videoActive = true;
  };

  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
  };

  socket.onclose = () => {
    console.log("WebSocket closed");
  };

  socket.onmessage = function (event) {
    if (videoActive) {
      const frameBase64 = event.data;
      const blob = new Blob([frameBase64], { type: "image/jpeg" });
      const imageUrl = URL.createObjectURL(blob);

      const img = new Image();
      img.onload = function () {
        ctx.drawImage(img, 0, 0, videoCanvas.width, videoCanvas.height);
      };
      img.src = imageUrl;
      const data = JSON.parse(event.data);
      updateUI(data);
    }
  };
}

function deactivatevideo() {
  videoActive = false;
  const request = { command: "deactivate_video" };
  socket.send(JSON.stringify(request));

  const videoCanvas = document.getElementById("video-canvas");
  const ctx = videoCanvas.getContext("2d");
  ctx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
}

const videoButton = document.getElementById("videoButton");
videoButton.addEventListener("click", () => {
  if (!videoActive) {
    activatevideo();
    videoButton.textContent = "Turn Off video";
  } else {
    deactivatevideo();
    videoButton.textContent = "Turn On video";
  }
});

function activateInput() {
  document.getElementById("fileInput").click();
}
