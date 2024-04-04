import * as React from "react";

import Head from "next/head";
import styles from "@/styles/styles.module.css";
import { getIronSession } from "iron-session";

import { NextUIProvider, Text } from "@nextui-org/react";
const { NavNotLogged, Nav } = require("@/components/Nav");

export default function about({ logged }) {
  if (logged) {
    return (
      <NextUIProvider>
        <Nav />
        <Head>
          <title>About</title>
        </Head>

        <Text className={styles.sectionTitleStyle}>
          If you need help use this program type "help me please"
        </Text>
      </NextUIProvider>
    );
  } else {
    return (
      <NextUIProvider>
        <NavNotLogged />
        <Head>
          <title>About</title>
        </Head>

        <Text className={styles.sectionTitleStyle}>
          Orpheus is Chatbot Music Recommender, Sign Up or Login for more
          information
        </Text>
      </NextUIProvider>
    );
  }
}

export async function getServerSideProps({ req, res }) {
  const session = await getIronSession(req, res, {
    cookieName: process.env.COOKIE_NAME,
    password: process.env.SESSION_PASSWORD,
  });
  console.log(process.env.API_ENDPOINT);
  var logged;
  if (session.userID == undefined) {
    return {
      props: {
        logged: false,
      },
    };
  }

  return {
    props: {
      logged: true,
    },
  };
}
