import { useEffect } from "react";
const { app, database } = require("@/components/firebase");
import { getDatabase, ref, child, get, remove } from "firebase/database";
import { getIronSession } from "iron-session";

import { NextUIProvider } from "@nextui-org/react";
import { Button, Link, Table, TableColumn, Spacer } from "@nextui-org/react";

const { Nav } = require("@/components/Nav");
export default function Profile({ userID, email, songsData, headers }) {
  async function removeSong(id) {
    await get(ref(database, "users/" + userID + "/" + id))
      .then((snapshot) => {
        if (snapshot.exists()) {
          remove(ref(database, "users/" + userID + "/" + id));

          console.log("sss'" + id + "'");
        } else {
        }
      })
      .catch((error) => {
        console.error(error);
      });
  }

  return (
    <NextUIProvider>
      <Nav />

      {/* {email ? <p>This is you email: {email}</p> : <p>Loading...</p>}
      {JSON.stringify(songsData)} */}

      <table>
        <thead>
          <tr>
            {headers.map((header) => (
              <th key={header} style={{ fontSize: "10px" }}>
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {songsData.map((item, index) => (
            <tr key={index}>
              {headers.map((header) => (
                <td key={header}>
                  {JSON.parse(Object.values(item)[1])[0][header]}
                </td>
              ))}
              <td>
                <Button onPress={removeSong(Object.values(item)[0])}>
                  Delete
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </NextUIProvider>
  );
}

export async function getServerSideProps({ req, res }) {
  const session = await getIronSession(req, res, {
    cookieName: process.env.COOKIE_NAME,
    password: process.env.SESSION_PASSWORD,
  });
  console.log("-------------------------");

  if (session.userID == undefined) {
    return {
      redirect: {
        permanent: false,
        destination: "login",
      },
    };
  }

  var songs;
  const dbRef = ref(database);
  await get(child(dbRef, `users/${session.userID}`))
    .then((snapshot) => {
      if (snapshot.exists()) {
        songs = snapshot.toJSON();
      } else {
        console.log("No data available");
        songs = "Start Uploading";
      }
    })
    .catch((error) => {
      console.error(error);
    });

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
  var headers = Object.keys(JSON.parse(Object.values(songs)[0])[0]);

  excludedKeys.forEach((element) => {
    headers = headers.filter((item) => item !== element);
  });
  var songsData = Object.entries(songs);

  // const dataObject = Object.entries(songs).map((item, index) => {
  //   var j = JSON.parse(Object.values(item)[1])[0];
  //   console.log(JSON.parse(Object.values(item)[1])[0]["chroma_stft_mean"]);
  // });

  // console.log(Object.values(JSON.parse(Object.values(songs)[1])[0]));

  return {
    props: {
      email: session.email,
      email: session.userID,
      songsData: songsData,
      headers: headers,
    },
  };
}
