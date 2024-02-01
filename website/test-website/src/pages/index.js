import { getIronSession } from "iron-session";
const { Nav } = require("@/components/Nav");
import React, { useState, useEffect } from "react";
const { Orpheus } = require("@/components/Orpheus");
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
