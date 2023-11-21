import { useState } from "react";

const Orpheus = () => {
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

  return (
    <div>
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
      <p>features: {features}</p>
      <p>songs : {songs}</p>
      <p>Spotify : {spotifySong}</p>
    </div>
  );
};

export default Orpheus;
