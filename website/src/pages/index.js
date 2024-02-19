import { getIronSession } from "iron-session";
import React from "react";
import { NextUIProvider } from "@nextui-org/react";
const { Orpheus } = require("@/components/Orpheus");
const { Nav } = require("@/components/Nav");
import Head from "next/head";

export default function Home({ userID, endpoint }) {
  return (
    <NextUIProvider>
      <Head>
        <title>Orpheus</title>
      </Head>

      <Nav />

      <Orpheus userID={userID} endpoint={endpoint} />
    </NextUIProvider>
  );
}

export async function getServerSideProps({ req, res }) {
  const session = await getIronSession(req, res, {
    cookieName: process.env.COOKIE_NAME,
    password: process.env.SESSION_PASSWORD,
  });
  console.log(process.env.API_ENDPOINT);

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
      endpoint: process.env.API_ENDPOINT,
    },
  };
}
