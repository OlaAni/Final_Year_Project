import React from "react";

import { useState } from "react";

import styles from "../../public/styles.css";

function Orpheus() {
  const [userInput, setUserInput] = useState("");
  const [response, setResponse] = useState("");
  const [confidence, setConfidence] = useState("");
  const [features, setFeatures] = useState("");
  const [songs, setSongs] = useState("");
  const [spotifySong, setSpotifySong] = useState("");

  const handleUserInput = (event) => {
    setUserInput(event.target.value);
  };

  async function handleSubmit(event) {
    event.preventDefault();
    const url = "http://localhost:5000/chat";

    var options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ user_input: userInput }),
    };

    if (!features == null || !features.trim().length == 0) {
      options.body = JSON.stringify({
        user_input: userInput,
        features: features,
      });
    }

    const response = await fetch(url, options);
    const result = await response.json();

    // alert(result.status);
    if (result.status == "OK") {
      setResponse(result.Orpheus);

      if (result.confidence != null) {
        setConfidence(result.confidence);
      }

      if (result.features != null) {
        setFeatures(result.features);
      }

      if (result.songs != null) {
        setSongs(result.songs);
      }

      if (result.recommendation != null) {
        setSpotifySong(result.recommendation);
      }
    } else {
      alert("Status: " + result.status);
    }
  }

  async function uploadFile(event) {
    const url = "http://localhost:5000/chat";
    const file = event.target.files[0];

    const formData = new FormData();

    formData.append("user_input", "extract");
    formData.append("music_file", file);

    var options = {
      method: "POST",
      body: formData,
    };

    const response = await fetch(url, options);
    const result = await response.json();

    if (result.status == "OK") {
      setResponse(result.Orpheus);

      if (result.confidence != null) {
        setConfidence(result.confidence);
      }

      if (result.features != null) {
        setFeatures(result.features);
      }
    }
  }

  return (
    <>
      <div className="container">
        <div className="column">
          <p>features: {features}</p>
          <p>songs : {songs}</p>
          <p>Spotify : {spotifySong}</p>
        </div>

        <div className="column">
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              value={userInput}
              id="user_input"
              onChange={handleUserInput}
            />
            <button>Chat</button>
          </form>

          <p>Orpheus: {response}</p>
          <p>Confidence: {confidence}</p>
        </div>

        <div className="column">
          <input type="file" onChange={uploadFile} />
        </div>
      </div>
    </>
  );
}

// export default Orpheus;
module.exports = {
  Orpheus,
};
