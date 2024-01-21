import { useEffect } from "react";
const { app, database } = require("@/components/firebase");
import { getDatabase, ref, child, get } from "firebase/database";
import { getIronSession } from "iron-session";

import { NextUIProvider } from "@nextui-org/react";
import { Button, Link, Table, TableColumn, Spacer } from "@nextui-org/react";

const { Nav } = require("@/components/Nav");
export default function Profile({ email, songsData }) {
  return (
    <NextUIProvider>
      <Nav />

      {email ? <p>This is you email: {email}</p> : <p>Loading...</p>}
      {JSON.stringify(songsData)}
    </NextUIProvider>
  );
}

export async function getServerSideProps({ req, res }) {
  const session = await getIronSession(req, res, {
    cookieName: process.env.COOKIE_NAME,
    password: process.env.SESSION_PASSWORD,
  });
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

  console.log(songs);

  return {
    props: {
      email: session.email,
      songsData: songs,
    },
  };
}
