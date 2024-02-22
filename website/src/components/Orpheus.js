import { getIronSession } from "iron-session";
import React, { useState, useEffect } from "react";
import { ref, push, set } from "firebase/database";
import { ref as refStorage, getDownloadURL, listAll } from "firebase/storage";

const { app, database, storage } = require("@/components/firebase");

import { NextUIProvider, yellow } from "@nextui-org/react";
import {
  Button,
  Link,
  Grid,
  Spacer,
  Text,
  Switch,
  Popover,
} from "@nextui-org/react";
import { color } from "framer-motion";
import styles from "@/styles/styles.module.css";

function Orpheus({ userID, endpoint }) {
  const [userInput, setUserInput] = useState("");
  const [confidence, setConfidence] = useState([""]);
  const [features, setFeatures] = useState("");
  const [songs, setSongs] = useState([]);
  const [spotifySong, setSpotifySong] = useState("");
  const [messages, setMessages] = useState([""]);
  const [isLoading, setIsLoading] = useState(false);
  const [isChecked, setisChecked] = useState(false);

  function getRandomInt(max) {
    return Math.floor(Math.random() * max + 10);
  }

  const ChatWindow = ({ messages }) => {
    return (
      <div className={styles.chatWindowStyle}>
        {isLoading ? (
          <div>Loading...</div>
        ) : (
          <div className={styles.messageContainerStyle}>
            {messages.map((message, index) => (
              <div key={index} className={styles.messageStyle}>
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
    console.log(song);
    var folder = song[0].match(/([a-zA-Z]+)/);
    var filename = folder[0] + "/" + song[0];
    // console.log(filename);

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
    const url = endpoint + "/chat";
    addMessage("You: " + userInput);

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
    setIsLoading(true);

    try {
      const response = await fetch(url, options);
      const result = await response.json();
      if (result.status == "OK") {
        if (result.Orpheus.includes("give_me_a_song")) {
          const match = result.Orpheus.match(/(\w+)\?/);
          result.Orpheus = result.Orpheus.replace("give_me_a_song", "");

          var num = getRandomInt(89);
          var song = [match[1].toLowerCase() + ".000" + num + ".wav", 1];
          var newLink = await getDownloadLink(song);
          addMessage(
            <span>
              Orpheus: {result.Orpheus} Check{" "}
              <Link href={newLink} target="_blank">
                here
              </Link>
              !!!
            </span>
          );
        } else {
          addMessage("Orpheus: " + result.Orpheus);
        }

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
    } catch (error) {
      addMessage(
        "Orpheus: Orpheus Prime seeems to indiposed for a second, try again in a minute"
      );

      setIsLoading(false);
    }
  }

  async function uploadFile(event) {
    const url = endpoint + "/upload";
    console.log(url);

    const file = event.target.files[0];

    if (event.target.files[0].size > 1500000) {
      // alert(event.target.files[0].size);
      addMessage(
        "Orpheus: I can only handle so much data, gonna need you to slim it down"
      );
    } else {
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
  }

  // const jsonData =
  //   '[{"chroma_stft_mean":0.2914259732,"chroma_stft_var":0.0939848498,"harmony_mean":-0.0000117495,"harmony_var":0.0169314761,"rms_mean":0.1417519599,"rms_var":0.0087820757,"rolloff_mean":2795.910729499,"rolloff_var":1621507.8096568789,"spectral_bandwidth_mean":1389.0684455566,"spectral_bandwidth_var":185156.3212288567,"spectral_centroid_mean":1459.4696908115,"spectral_centroid_var":438024.9988025444,"tempo":123.046875,"zero_crossing_rate_mean":0.0730522374,"zero_crossing_rate_var":0.0019091707,"label":0}]';

  const jsonData = '[{"Example":"Characteristics"}]';
  if (!Object.keys(features).length) {
    var features1 = JSON.parse(jsonData)[0];
  } else {
    var features1 = JSON.parse(features)[0];
    features1["label"] = confidence[0][0];
  }

  const excludedKeys = [
    "chroma_stft_var",
    "harmony_var",
    "rolloff_var",
    "spectral_bandwidth_var",
    "spectral_centroid_var",
    "zero_crossing_rate_var",
    "rms_var",
    "filename",
  ];

  var originalKeys = Object.keys(features1);

  originalKeys = originalKeys.filter((key) => !excludedKeys.includes(key));

  const activeKeys = isChecked
    ? [
        "Pitch",
        "Harmony",
        "Loudness",
        "Energy",
        "Sporadicity",
        "Brightness",
        "Tempo",
        "Beats",
        "Genre",
      ]
    : originalKeys;

  function changeKey(key) {
    if (key == "Pitch") return "chroma_stft_mean";
    else if (key == "Harmony") return "harmony_mean";
    else if (key == "Loudness") return "rms_mean";
    else if (key == "Energy") return "rolloff_mean";
    else if (key == "Sporadicity") return "spectral_bandwidth_mean";
    else if (key == "Brightness") return "spectral_centroid_mean";
    else if (key == "Tempo") return "tempo";
    else if (key == "Beats") return "zero_crossing_rate_mean";
    else if (key == "Genre") return "label";
    else {
      return key;
    }
  }

  function displayMeaning(key) {
    if (key == "Pitch" || key == "chroma_stft_mean")
      return "High values usually denote metal or hiphop, low values denote classical or jazz";
    else if (key == "Harmony" || key == "harmony_mean")
      return "High values usually denote hiphop and pop, low values metal and rock";
    else if (key == "Loudness" || key == "rms_mean")
      return "High values usually denote hiphop and pop, low values reggae and jazz";
    else if (key == "Energy" || key == "rolloff_mean")
      return "High values usually denot pop, low values blues and classical";
    else if (key == "Sporadicity" || key == "spectral_bandwidth_mean")
      return "High values usually denote disco and pop, low values country and classical";
    else if (key == "Brightness" || key == "spectral_centroid_mean")
      return "High values usually denote disco and pop, low values classical and blues";
    else if (key == "Tempo" || key == "tempo")
      return "Songs dont tend to differ";
    else if (key == "Beats" || key == "zero_crossing_rate_mean")
      return "High values usually denote pop and jazz, low values jazz";
    else if (key == "label" || key == "Genre") return "How right was I???";
    else {
      return "Upload some songs!!!";
    }
  }
  return (
    <NextUIProvider>
      <Grid.Container gap={2} justify="center">
        <Grid xs={3} direction="column" className={styles.columnStyle}>
          <div className={styles.centeredContainer}>
            <Switch
              checked={isChecked}
              onChange={() => setisChecked(!isChecked)}
              style={{ color: "#daa520" }}
              color="warning"
              size="md"
            />
          </div>

          <Text className={styles.sectionTitleStyle}>
            {isChecked ? "Standard" : "Scientific"}
          </Text>
          <Text className={styles.sectionTitleStyle}>Features</Text>
          <table>
            <thead>
              <tr>
                <th>Key</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {activeKeys.map((key, index) => (
                <tr key={index}>
                  <td>
                    <Popover placement={"left"}>
                      <Popover.Trigger>
                        <Button auto bordered color="secondary">
                          {key}
                        </Button>
                      </Popover.Trigger>
                      <Popover.Content>
                        <Text css={{ p: "$10" }}>{displayMeaning(key)}</Text>
                      </Popover.Content>
                    </Popover>
                  </td>
                  <td>{JSON.stringify(features1[changeKey(key)])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Grid>
        <Grid xs={6} direction="column" className={styles.columnStyle}>
          <ChatWindow messages={messages} isLoading={isLoading} />
          <Spacer y={3} />
          <input
            type="text"
            value={userInput}
            id="user_input"
            onChange={handleUserInput}
            className={styles.inputStyle}
          />
          <Button color="warning" onPress={handleSubmit}>
            Chat
          </Button>
        </Grid>
        <Grid xs={2} direction="column" className={styles.columnStyle}>
          <input
            type="file"
            onChange={uploadFile}
            className={styles.fileInputStyle}
          />
          <Spacer y={3} />
          <Text className={styles.sectionTitleStyle}>Confidence Breakdown</Text>
          <div className={styles.centeredContainer}>
            <ConfidenceScores scores={confidence} />
          </div>
          <Spacer y={3} />

          <Text className={styles.sectionTitleStyle}>Songs</Text>
          <RecoSongs songs={songs} />

          <Spacer y={3} />
          <Text className={styles.sectionTitleStyle}>Spotify</Text>
          <Text style={{ overflowWrap: "break-word" }}>{spotifySong}</Text>
        </Grid>
      </Grid.Container>
    </NextUIProvider>
  );
}

module.exports = {
  Orpheus,
};
