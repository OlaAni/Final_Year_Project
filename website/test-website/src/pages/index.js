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
