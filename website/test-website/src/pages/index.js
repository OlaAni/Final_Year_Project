// const { Orpheus } = require("@/pages/test.js");
import { getIronSession } from "iron-session";
const { Nav } = require("@/components/Nav");
import React, { useState, useEffect } from "react";
import { ref, push, set } from "firebase/database";
import { ref as refStorage, getDownloadURL } from "firebase/storage";

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

export default function Home({ userID }) {
  const [userInput, setUserInput] = useState("");
  const [confidence, setConfidence] = useState([""]);
  const [features, setFeatures] = useState("");
  const [songs, setSongs] = useState([]);
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
    // console.log(folder[0] + " folder, filename: " + song[0]);
    try {
      const url = await getDownloadURL(
        refStorage(storage, folder[0] + "/" + song[0])
      );
      // console.log(url);
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
    const url = "http://localhost:5000/chat";
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
      <Nav />

      <Grid.Container gap={2} justify="center">
        <Grid xs={4} direction="column">
          <Text>Features</Text>
          <Text style={{ overflowWrap: "break-word" }}>{features}</Text>
          <Spacer y={3} />
          <Text>Songs</Text>
          {/* <Text style={{ overflowWrap: "break-word" }}>{songs} </Text> */}
          <RecoSongs songs={songs} />
          <Spacer y={3} />
          <Text>Spotify</Text>
          <Text style={{ overflowWrap: "break-word" }}>{spotifySong}</Text>
        </Grid>
        <Grid xs={5} direction="column">
          <ChatWindow messages={messages} isLoading={isLoading} />
          <Spacer y={3} />
          <form>
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
        <Grid xs={2} direction="column">
          <input type="file" onChange={uploadFile} />
          <Spacer y={3} />
          <Text>Confidence Breakdown</Text>
          <ConfidenceScores scores={confidence} />
        </Grid>
      </Grid.Container>

      {/* <Orpheus userID={userID} /> */}
    </NextUIProvider>
  );
}

export async function getServerSideProps({ req, res }) {
  const session = await getIronSession(req, res, {
    cookieName: process.env.COOKIE_NAME,
    password: process.env.SESSION_PASSWORD,
  });

  if (session.userID == undefined) {
    return {
      redirect: {
        permanent: false,
        destination: "login",
      },
    };
  }

  return {
    props: {
      email: session.email,
      userID: session.userID,
    },
  };
}
