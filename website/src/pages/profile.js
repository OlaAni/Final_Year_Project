import React, { useState } from "react";
const { app, database } = require("@/components/firebase");
import { getDatabase, ref, child, get, remove } from "firebase/database";
import { getIronSession } from "iron-session";

import { NextUIProvider } from "@nextui-org/react";
import {
  Button,
  Link,
  Table,
  TableColumn,
  Spacer,
  Text,
  Switch,
  Popover,
} from "@nextui-org/react";

const { Nav } = require("@/components/Nav");
import Head from "next/head";
import styles from "@/styles/styles.module.css";

export default function Profile({ userID, songsData, headers }) {
  const [tableData, setTableData] = useState(songsData);
  const [isChecked, setisChecked] = useState(true);

  const removeSong = (userId, id) => {
    console.log(userId + " : " + id);
    get(ref(database, "users/" + userId + "/" + id))
      .then((snapshot) => {
        if (snapshot.exists()) {
          remove(ref(database, "users/" + userId + "/" + id));
          setTableData(
            tableData.filter((item) => Object.values(item)[0] !== id)
          );
        }
      })
      .catch((error) => {
        console.error(error);
      });
  };
  function changeKey(key) {
    if (key == "filename") return "Filename";
    if (isChecked) {
      if (key == "chroma_stft_mean") return "Pitch";
      else if (key == "harmony_mean") return "Harmony";
      else if (key == "rms_mean") return "Loudness";
      else if (key == "rolloff_mean") return "Energy";
      else if (key == "spectral_bandwidth_mean") return "Sporadicity";
      else if (key == "spectral_centroid_mean") return "Brightness";
      else if (key == "tempo") return "Tempo";
      else if (key == "zero_crossing_rate_mean") return "Beats";
      else if (key == "label") return "Genre";
    } else {
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

  function convertGenre(key, value) {
    if (key == "label" || key == "Genre") {
      if (value == "0") return "blues";
      else if (value == "1") return "classical";
      else if (value == "2") return "country";
      else if (value == "3") return "disco";
      else if (value == "4") return "hiphop";
      else if (value == "5") return "jazz";
      else if (value == "6") return "metal";
      else if (value == "7") return "pop";
      else if (value == "8") return "reggae";
      else if (value == "9") return "rock";
    } else {
      if (isChecked && !(key == "filename" || key == "Filename")) {
        if (key == "Harmony" || key == "harmony_mean") {
          return parseFloat(value).toFixed(6);
        } else {
          return parseFloat(value).toFixed(3);
        }
      } else {
        return value;
      }
    }
  }

  return (
    <NextUIProvider>
      <Head>
        <title>Profile</title>
      </Head>
      <Nav />
      {/* {headers ? <p>This is you id: {headers}</p> : <p>Loading...</p>} */}
      <div className={styles.sectionTitleStyle}>
        <Switch
          checked={isChecked}
          onChange={() => setisChecked(!isChecked)}
          style={{ color: "#daa520", backgroundColor: "#daa520" }}
          color="warning"
          size="md"
        />
      </div>
      <Text className={styles.sectionTitleStyle}>
        {isChecked ? "Standard" : "Scientific"}
      </Text>
      <div style={{ display: "flex", justifyContent: "center" }}>
        {songsData === "no" ? (
          <Text
            css={{ fontSize: "40px", fontWeight: "bold", marginBottom: "10px" }}
          >
            Upload some Songs!!!
          </Text>
        ) : (
          <table>
            <thead>
              <tr>
                {headers.map((header) => (
                  <th
                    key={header}
                    style={{
                      fontSize: "10px",
                      backgroundColor: "white",
                      border: "2px solid #ddd",
                    }}
                  >
                    <Popover placement={"top"}>
                      <Popover.Trigger>
                        <Button auto bordered color="secondary">
                          {changeKey(header)}
                        </Button>
                      </Popover.Trigger>
                      <Popover.Content>
                        <Text css={{ p: "$10" }}>{displayMeaning(header)}</Text>
                      </Popover.Content>
                    </Popover>
                    {/* {changeKey(header)} */}
                  </th>
                ))}
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {tableData.map((item, index) => (
                <tr key={index}>
                  {headers.map((header) => (
                    <td key={header}>
                      {convertGenre(
                        header,
                        JSON.parse(Object.values(item)[1])[0][header]
                      )}
                    </td>
                  ))}
                  <td
                    style={{
                      border: "5px solid #ddd",
                    }}
                  >
                    <Button
                      onClick={() => removeSong(userID, Object.values(item)[0])}
                      style={{ background: "black" }}
                    >
                      Delete
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
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

  const excludedKeys = [
    "chroma_stft_var",
    "harmony_var",
    "harmony_var",
    "rolloff_var",
    "spectral_bandwidth_var",
    "spectral_centroid_var",
    "zero_crossing_rate_var",
    "rms_var",
  ];
  var songsData;
  var headers;

  const dbRef = ref(database);
  await get(child(dbRef, `users/${session.userID}`))
    .then((snapshot) => {
      if (snapshot.exists()) {
        songsData = snapshot.toJSON();

        headers = Object.keys(JSON.parse(Object.values(songsData)[0])[0]);

        excludedKeys.forEach((element) => {
          headers = headers.filter((item) => item !== element);
        });
        songsData = Object.entries(songsData);
      } else {
        console.log("No data available");
        songsData = "no";
        headers = "no";
      }
    })
    .catch((error) => {
      console.error(error);
    });

  return {
    props: {
      userID: session.userID,
      songsData: songsData,
      headers: headers,
    },
  };
}
