import { getIronSession } from "iron-session";
import React, { useState, useEffect } from "react";
import { ref, push, set } from "firebase/database";
import { ref as refStorage, getDownloadURL, listAll } from "firebase/storage";

const { app, database, storage } = require("@/components/firebase");

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


function Orpheus({ userID, endpoint }) {
  const [userInput, setUserInput] = useState("");
  const [confidence, setConfidence] = useState([""]);
  const [features, setFeatures] = useState("");
  const [songs, setSongs] = useState([]);
  const [spotifySong, setSpotifySong] = useState("");
  const [messages, setMessages] = useState([""]);
  const [isLoading, setIsLoading] = useState(false);

  const ChatWindow = ({ messages }) => {
    return (
      <div style={chatWindowStyle}>
        {isLoading ? (
          <div>Loading...</div>
        ) : (
          <div style={messageContainerStyle}>
            {messages.map((message, index) => (
              <div key={index} style={messageStyle}>
                {message}
              </div>
            ))}
          </div>
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
          scores.map((score, index) => (
            <div key={index}>
              {score[0]} : {score[1]}%
            </div>
          ))
        )}
      </div>
    );
  };

  const RecoSongs = ({ songs }) => {
    const [recos, setRecos] = useState([]);

    useEffect(() => {
      const fetchData = async () => {
        const newRecos = [];

        await Promise.all(
          songs.map(async (song) => {
            const newEntry = {
              name: song[0],
              link: await getDownloadLink(song),
              sim: song[1],
            };
            newRecos.push(newEntry);
          })
        );

        setRecos(newRecos);
      };

      fetchData();
    }, [songs]);

    return recos.map((newEntry, index) => {
      if (recos.length < 1) {
        return <div>Null</div>;
      } else {
        return (
          <div>
            {
              <div key={index}>
                <Link href={newEntry.link} target="_blank">
                  {newEntry.name}
                </Link>{" "}
                : {parseFloat(newEntry.sim).toFixed(2)}%
              </div>
            }
          </div>
        );
      }
    });
  };
  async function getDownloadLink(song) {
    var folder = song[0].match(/([a-zA-Z]+)/);
    var filename = folder[0] + "/" + song[0];
    console.log(filename);

    try {
      const url = await getDownloadURL(refStorage(storage, filename));
      console.log(filename + ":" + url);
      return url;
    } catch (error) {
      console.error("Error fetching download link:", error);
      return "default";
    }
  }
  const addMessage = (newMessage) => {
    setMessages((prevMessages) => [...prevMessages, newMessage]);
  };

  const handleUserInput = (event) => {
    setUserInput(event.target.value);
  };

  async function handleSubmit(event) {
    // event.preventDefault();
    const url = endpoint+"/chat";

    console.log(endpoint)

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

    addMessage("You: " + userInput);
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
        console.log(result.songs[0]);
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
    const url = endpoint+"/chat";
    console.log(url)

    const file = event.target.files[0];
    console.log(userID);

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
        <Grid xs={4} direction="column" style={columnStyle}>
          <Text style={sectionTitleStyle}>Features</Text>
          <Text style={breakWordStyle}>{features}</Text>
          <Spacer y={3} />
          <Text style={sectionTitleStyle}>Songs</Text>
          <RecoSongs songs={songs} />
          <Spacer y={3} />
          <Text style={sectionTitleStyle}>Spotify</Text>
          <Text style={breakWordStyle}>{spotifySong}</Text>
        </Grid>
        <Grid xs={5} direction="column" style={columnStyle}>
          <ChatWindow messages={messages} isLoading={isLoading} />
          <Spacer y={3} />
          <form style={formStyle}>
            <input
              type="text"
              value={userInput}
              id="user_input"
              onChange={handleUserInput}
              style={inputStyle}
            />
            <Button color="primary" onPress={handleSubmit}>
              Chat
            </Button>
          </form>
        </Grid>
        <Grid xs={2} direction="column" style={columnStyle}>
          <input type="file" onChange={uploadFile} style={fileInputStyle} />
          <Spacer y={3} />
          <Text style={sectionTitleStyle}>Confidence Breakdown</Text>
          <ConfidenceScores scores={confidence} />
        </Grid>
      </Grid.Container>
    </NextUIProvider>
  );
}

const columnStyle = {
  padding: "20px",
  border: "1px solid #ddd",
  borderRadius: "8px",
  background: "#fff",
};

const sectionTitleStyle = {
  fontSize: "18px",
  fontWeight: "bold",
  marginBottom: "10px",
};

const breakWordStyle = {
  overflowWrap: "break-word",
};

const formStyle = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  background: "black",
};

const inputStyle = {
  marginBottom: "10px",
  padding: "8px",
  border: "1px solid #ddd",
  borderRadius: "4px",
};

const fileInputStyle = {
  marginBottom: "10px",
};
const chatWindowStyle = {
  border: "1px solid #ddd",
  borderRadius: "8px",
  minHeight: "300px",
  maxHeight: "500px",
  overflowY: "auto",
  padding: "10px",
};

const messageContainerStyle = {
  overflowY: "auto",
};

const messageStyle = {
  marginBottom: "8px",
  padding: "8px",
  border: "1px solid #ddd",
  borderRadius: "4px",
};
module.exports = {
  Orpheus,
};
