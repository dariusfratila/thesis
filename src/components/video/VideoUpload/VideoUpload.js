import React, { useState, useCallback, useRef } from "react";
import axios from "axios";
import styles from "./VideoUpload.module.css";
import TextToSpeech from "../../speech/TextToSpeech/TextToSpeech";
import PredictionsDisplay from "../../speech/PredictionsDisplay/PredictionsDisplay";
import { ScrollRestoration } from "react-router-dom";
import ScrollToExplore from "../../common/ScrollToExplore/ScrollToExplore";
import ScrollToPredictions from "../../common/ScrollToPredictions/ScrollToPredictions";
import VideoCapture from "../VideoCapture/VideoCapture";

function VideoUpload() {
  const [videoFile, setVideoFile] = useState(null);
  const [stream, setStream] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [predictionResults, setPredictionResults] = useState({
    words: [],
    probabilities: [],
  });
  const [gifUrl, setGifUrl] = useState("");
  const videoRef = useRef();
  const mediaRecorderRef = useRef();
  const predictionsRef = useRef(null);

  const WaveSVG = () => (
    <svg
      className={styles.svgWave}
      viewBox="0 0 500 100"
      preserveAspectRatio="none"
    >
      <path
        d="M-0.00,39.98 C249.99,100.00 149.20,-19.98 600.00,59.98 L500.00,150.00 L0.00,150.00 Z"
        style={{ stroke: "none", fill: "black" }}
      ></path>
    </svg>
  );

  const startVideoStream = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });
      setStream(mediaStream);
      videoRef.current.srcObject = mediaStream;
    } catch (err) {
      console.error("Error accessing the camera: ", err);
    }
  };

  const startRecording = () => {
    if (!stream) return;
    mediaRecorderRef.current = new MediaRecorder(stream);
    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data.size > 0) {
        const newBlob = new Blob([event.data], { type: "video/webm" });
        setVideoFile(newBlob);
      }
    };
    mediaRecorderRef.current.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setIsRecording(false);
    setStream(null);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const handleVideoChange = useCallback((event) => {
    const files = event.target.files || event.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith("video/")) {
        setVideoFile(file);
        setPredictionResults({ words: [], probabilities: [] });
        setIsUploading(false);
      } else {
        alert("Please upload a video file.");
      }
    }
  }, []);

  const uploadVideo = useCallback(() => {
    if (!videoFile) {
      console.log("No video file selected.");
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", videoFile);

    const config = {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    };

    axios
      .post("http://127.0.0.1:5000/demo", formData, config)
      .then((response) => {
        alert("Video uploaded successfully!");

        const gifPath = response.data.saliency_maps_gif;
        const url = `http://127.0.0.1:5000/images/${gifPath.replace(
          /^.*\/uploaded_videos\//,
          ""
        )}`;
        setGifUrl(url);
        console.log("GIF URL Set to:", url);

        if (response.data.predictions) {
          const words = response.data.predictions.map(
            (prediction) => prediction[0]
          );
          const probabilities = response.data.predictions.map(
            (prediction) => prediction[1]
          );
          setPredictionResults({ words, probabilities });
        } else {
          setPredictionResults({ words: [], probabilities: [] });
        }
      })
      .catch((error) => {
        alert(`Failed to upload video: ${error.message}`);
      })
      .finally(() => {
        setIsUploading(false);
      });
  }, [videoFile]);

  const clearVideo = useCallback(() => {
    setVideoFile(null);
    setPredictionResults({ words: [], probabilities: [] });
    setIsUploading(false);
  }, []);

  const handleDragEvents = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(event.type === "dragover");
  }, []);

  const handleDrop = useCallback(
    (event) => {
      handleDragEvents(event);
      handleVideoChange(event);
    },
    [handleDragEvents, handleVideoChange]
  );

  const UploadPrompt = () => (
    <div className={styles.uploadPrompt}>
      <i className="fas fa-upload fa-2x"></i>
      <input
        id="video-upload"
        type="file"
        onChange={handleVideoChange}
        accept="video/*"
        className={styles.uploadInput}
      />
      <label htmlFor="video-upload">Choose a video or drag it here.</label>
    </div>
  );

  // const handleScrollToPredictions = () => {
  //   console.log("Attempting to scroll to predictions");
  //   if (predictionsRef.current) {
  //     predictionsRef.current.scrollIntoView({ behavior: "smooth" });
  //   }
  // };

  const VideoPreview = () => (
    <div className={styles.videoPreviewWrapper}>
      <video
        controls
        src={URL.createObjectURL(videoFile)}
        className={styles.videoPreview}
      />
      <div className={styles.videoActions}>
        <button
          onClick={uploadVideo}
          className={styles.uploadButton}
          disabled={isUploading}
        >
          {isUploading ? "Uploading..." : "Upload Video"}
        </button>
        <button
          onClick={clearVideo}
          className={styles.clearButton}
          disabled={isUploading}
        >
          Clear Video
        </button>
      </div>
    </div>
  );

  return (
    <div className={styles.container}>
      <div className={styles.videoUploadSection}>
        <div className={styles.header}>
          <h1>Ready, Set, Upload!</h1>
          <p>Drag & Drop or Click to Upload Your Video</p>
        </div>
        <div
          className={`${styles.uploadArea} ${
            dragActive ? styles.dragActive : ""
          } ${isUploading ? styles.uploadAreaAnimating : ""}`}
          onDragOver={handleDragEvents}
          onDragLeave={handleDragEvents}
          onDrop={handleDrop}
          tabIndex={0}
          aria-label="Upload area"
        >
          {!videoFile && <UploadPrompt />}
          {videoFile && <VideoPreview />}
          {isUploading && (
            <div className={styles.spinnerWrapper}>
              <div className={styles.uploadStatus}>
                <div className={styles.spinner}></div>
                <p className={styles.loadingText}>Uploading your video</p>
              </div>
            </div>
          )}
        </div>
        <div className={styles.rules}>
          <h2>Quick Tips:</h2>
          <ul>
            <li>Bright lighting, clear face.</li>
            <li>Your video, your rights.</li>
            <li>MP4 or MOV preferred.</li>
          </ul>
        </div>

        {predictionResults.words.length > 0 && (
          <>
            {/* <ScrollToPredictions onClick={handleScrollToPredictions} /> */}
            {/* <WaveSVG /> */}
            {/* <VideoCapture/> */}
            <PredictionsDisplay
              ref={predictionsRef}
              words={predictionResults.words}
              probabilities={predictionResults.probabilities}
            />

            {}
            {console.log(`GIF URL: ${gifUrl}`)}

            {gifUrl && (
              <div className={styles.gifDisplay}>
                <img src={gifUrl} alt="Saliency Maps GIF" />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default VideoUpload;
