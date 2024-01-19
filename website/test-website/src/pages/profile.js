import { useEffect } from "react";
const { app, database } = require("@/components/firebase");
import { getDatabase, ref, child, get } from "firebase/database";
import { getIronSession } from "iron-session";

import { NextUIProvider } from "@nextui-org/react";
import { Button, Link, Table, TableColumn } from "@nextui-org/react";

import { withIronSessionSsr } from "iron-session/next";

export default function Profile({ email, songsData }) {
  return (
    <NextUIProvider>
      <Link href="/">Orpheus</Link>
      <Button color="primary">Button</Button>
      {email ? <p>Welcome, {email}!</p> : <p>Loading...</p>}
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

// export const getServerSideProps = withIronSessionSsr(
//   async function getServerSideProps({ req }) {
//     var songs;
//     const dbRef = ref(database);

//     console.log(req.session.email);

//     await get(child(dbRef, `users/${req.session.userID}`))
//       .then((snapshot) => {
//         if (snapshot.exists()) {
//           console.log(snapshot.val());
//           //completedPuzzles = snapshot.toJSON();
//         } else {
//           console.log("No data available");
//         }
//       })
//       .catch((error) => {
//         console.error(error);
//       });

//     return {
//       props: {
//         // email: req.session.email,
//       },
//     };
//   }, // -------------------- All boilerplate code for sessions ------------------------------------
//   {
//     cookieName: process.env.COOKIE_NAME,
//     password: process.env.SESSION_PASSWORD,
//   }
// );
