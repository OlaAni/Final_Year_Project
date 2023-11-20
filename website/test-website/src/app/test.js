"use client";

import { useState } from "react";

const Orpheus = () => {
  const [userInput, setUserInput] = useState("");
  const [response, setResponse] = useState("");
  const [confidence, setConfidence] = useState("");

  const handleUserInput = (event) => {
    setUserInput(event.target.value);
  };

  async function handleSubmit(event) {
    event.preventDefault();
    const url = "http://localhost:5000/chat";

    const data = {
      username: event.target.user_input.value,
    };

    const options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ user_input: userInput }),
    };

    const response = await fetch(url, options);
    const result = await response.json();

    // alert(result.status);
    if (result.status == "OK") {
      setResponse(result.Orpheus);
      if (result.confidence != null) {
        setConfidence(result.confidence);
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
    </div>
  );
};

export default Orpheus;
