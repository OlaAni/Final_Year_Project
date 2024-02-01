import React, { useState } from "react";
const { app, database } = require("@/components/firebase");
import { getDatabase, ref, child, get, remove } from "firebase/database";
import { getIronSession } from "iron-session";

import { NextUIProvider } from "@nextui-org/react";
import { Button, Link, Table, TableColumn, Spacer } from "@nextui-org/react";

const { Nav } = require("@/components/Nav");

export default function Profile({ userID, songsData, headers }) {
  const [tableData, setTableData] = useState(songsData);
  const removeSong = (userId, id) => {
    console.log(userId + " : " + id);
    get(ref(database, "users/" + userId + "/" + id))
      .then((snapshot) => {
        if (snapshot.exists()) {
          remove(ref(database, "users/" + userId + "/" + id));
          setTableData(
            tableData.filter((item) => Object.values(item)[0] !== id)
          );

          console.log("sss'" + id + "'");
        } else {
        }
      })
      .catch((error) => {
        console.error(error);
      });
  };

  return (
    <NextUIProvider>
      <Nav />
      {/* {userID ? <p>This is you id: {userID}</p> : <p>Loading...</p>} */}
      <div style={{ display: "flex", justifyContent: "center" }}>
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
            {tableData.map((item, index) => (
              <tr key={index}>
                {headers.map((header) => (
                  <td key={header}>
                    {JSON.parse(Object.values(item)[1])[0][header]}
                  </td>
                ))}
                <td>
                  <Button
                    onClick={() => removeSong(userID, Object.values(item)[0])}
                  >
                    Delete
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
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
        songsData = "Start Uploading";
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
