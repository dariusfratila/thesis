import React, { useState, useEffect, useRef } from "react";

function VideoCapture() {
  const [stream, setStream] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const videoRef = useRef();
  const mediaRecorderRef = useRef();

  useEffect(() => {
    async function enableStream() {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        setStream(mediaStream);
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      } catch (err) {
        console.error("Failed to get video stream", err);
      }
    }

    if (!stream) {
      enableStream();
    } else {
      return () => {
        stream.getTracks().forEach((track) => track.stop());
      };
    }
  }, [stream]);

  const startRecording = () => {
    if (!stream) return;
    mediaRecorderRef.current = new MediaRecorder(stream, {
      mimeType: "video/webm",
    });
    mediaRecorderRef.current.addEventListener(
      "dataavailable",
      handleDataAvailable
    );
    mediaRecorderRef.current.start();
    setIsRecording(true);
  };

  const handleDataAvailable = (event) => {
    if (event.data.size > 0) {
      setRecordedChunks((prev) => prev.concat(event.data));
    }
  };

  const stopRecording = () => {
    if (!mediaRecorderRef.current) return;
    mediaRecorderRef.current.stop();
    setIsRecording(false);
  };

  const saveVideo = () => {
    if (recordedChunks.length) {
      const blob = new Blob(recordedChunks, { type: "video/webm" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "recorded-video.webm";
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        setRecordedChunks([]);
      }, 100);
    }
  };

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline />
      <div>
        {isRecording ? (
          <button onClick={stopRecording}>Stop Recording</button>
        ) : (
          <button onClick={startRecording}>Start Recording</button>
        )}
        <button onClick={saveVideo} disabled={!recordedChunks.length}>
          Save Video
        </button>
      </div>
    </div>
  );
}

export default VideoCapture;
