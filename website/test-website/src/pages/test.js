import React, { useState, useEffect } from "react";
import { ref, push, set } from "firebase/database";
const { app, database } = require("@/components/firebase");

import { NextUIProvider } from "@nextui-org/react";
import {
  Button,
  Link,
  Table,
  TableColumn,
  Grid,
  Spacer,
  Text,
} from "@nextui-org/react";

function Orpheus({ userID }) {
  const [userInput, setUserInput] = useState("");
  const [response, setResponse] = useState("");
  const [confidence, setConfidence] = useState([""]);
  const [features, setFeatures] = useState("");
  const [songs, setSongs] = useState("");
  const [spotifySong, setSpotifySong] = useState("");
  const [messages, setMessages] = useState([""]);
  const [isLoading, setIsLoading] = useState(false);

  const ChatWindow = ({ messages }) => {
    return (
      <div>
        {isLoading ? (
          <div>Loading...</div>
        ) : (
          messages.map((message, index) => <div key={index}>{message}</div>)
        )}
      </div>
    );
  };

  const ConfidenceScores = ({ scores }) => {
    return (
      <div>
        {isLoading ? (
          <div>Loading...</div>
        ) : (
          scores.map((message, index) => (
            <div key={index}>
              {message[0]} : {message[1]}%
            </div>
          ))
        )}
      </div>
    );
  };

  const addMessage = (newMessage) => {
    setMessages((prevMessages) => [...prevMessages, newMessage]);
  };

  const handleUserInput = (event) => {
    setUserInput(event.target.value);
  };

  async function handleSubmit(event) {
    // event.preventDefault();
    const url = "http://localhost:5000/chat";

    var options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ user_input: userInput, userID: userID }),
    };

    if (!features == null || !features.trim().length == 0) {
      options.body = JSON.stringify({
        user_input: userInput,
        features: features,
        userID: userID,
      });
    }

    addMessage("User: " + userInput);
    setIsLoading(true);

    const response = await fetch(url, options);
    const result = await response.json();

    // alert(result.status);
    if (result.status == "OK") {
      // setResponse(result.Orpheus);
      addMessage("Orpheus: " + result.Orpheus);

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
    setIsLoading(false);
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
    setIsLoading(true);

    const response = await fetch(url, options);
    const result = await response.json();

    if (result.status == "OK") {
      // setResponse(result.Orpheus);
      addMessage("Orpheus: " + result.Orpheus);

      if (result.confidence != null) {
        setConfidence(result.confidence);
        console.log(result.confidence);
      }

      if (result.features != null) {
        setFeatures(result.features);

        console.log(userID);
        const dataRef = ref(database, "users/" + userID);

        const newPushRef = push(dataRef);
        set(newPushRef, result.features)
          .then(() => {
            console.log("Data pushed successfully!");
          })
          .catch((error) => {
            console.error("Error pushing data:", error);
          });
      }
    }
    setIsLoading(false);
  }

  return (
    <NextUIProvider>
      <Grid.Container gap={2} justify="center">
        <Grid xs={4} direction="column">
          <Text>Features</Text>
          <Text style={{ overflowWrap: "break-word" }}>
            features: {features}
          </Text>
          <Spacer y={3} />
          <Text>Songs</Text>
          <Text style={{ overflowWrap: "break-word" }}>{songs} </Text>

          <Spacer y={3} />
          <Text>Spotify</Text>
          <Text style={{ overflowWrap: "break-word" }}>{spotifySong}</Text>
        </Grid>
        <Grid xs={4} direction="column">
          <ChatWindow messages={messages} isLoading={isLoading} />
          <Spacer y={3} />
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              value={userInput}
              id="user_input"
              onChange={handleUserInput}
            />
            <Button color="primary" onPress={handleSubmit}>
              Chat
            </Button>
          </form>
        </Grid>
        <Grid xs={4} direction="column">
          <input type="file" onChange={uploadFile} />
          <Spacer y={3} />
          <Text>Confidence Breakdown</Text>
          <ConfidenceScores scores={confidence} />
        </Grid>
      </Grid.Container>
    </NextUIProvider>
  );
}

// export default Orpheus;
module.exports = {
  Orpheus,
};
