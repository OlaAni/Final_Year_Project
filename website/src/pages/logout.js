import * as React from "react";

import { withIronSessionSsr } from "iron-session/next";

export default function Logout() {
  return (
    <>
      <h1>Logging out...</h1>
    </>
  );
}

export const getServerSideProps = withIronSessionSsr(
  async function getServerSideProps({ req }) {
    req.session.destroy();
    await req.session.save();
    return {
      redirect: {
        permanent: false,
        destination: "/login",
      },
    };
  }, // -------------------- All boilerplate code for sessions ------------------------------------
  {
    cookieName: process.env.COOKIE_NAME,
    password: process.env.SESSION_PASSWORD,
    cookieOptions: {
      secure: process.env.NODE_ENV === "production",
    },
  }
);
