"use client";
const { Orpheus } = require("@/pages/test.js");
import { NextUIProvider } from "@nextui-org/react";
import { getIronSession } from "iron-session";
const { Nav } = require("@/components/Nav");

export default function Home({ userID }) {
  return (
    <NextUIProvider>
      <Nav />
      <Orpheus userID={userID} />
    </NextUIProvider>
  );
}

export async function getServerSideProps({ req, res }) {
  const session = await getIronSession(req, res, {
    cookieName: process.env.COOKIE_NAME,
    password: process.env.SESSION_PASSWORD,
  });

  return {
    props: {
      email: session.email,
      userID: session.userID,
    },
  };
}
