import React from "react";

import { useState } from "react";
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
} from "@nextui-org/react";

function Orpheus({ userID }) {
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

    console.log(userID);

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
  }

  return (
    <NextUIProvider>
      <Grid.Container gap={2} justify="center">
        <Grid xs={4} direction="column">
          <p>features: {features}</p>
          <Spacer y={3} />
          <p>songs : {songs}</p>
          <Spacer y={3} />
          <p>Spotify : {spotifySong}</p>{" "}
        </Grid>
        <Grid xs={4} direction="column">
          {" "}
          <p>Orpheus: {response}</p>
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

          <p>Confidence: {confidence}</p>
        </Grid>
      </Grid.Container>
    </NextUIProvider>
  );
}

// export default Orpheus;
module.exports = {
  Orpheus,
};
