import React from "react";

import {
  Button,
  Link,
  Image,
  Navbar,
  Spacer,
  Text,
  Modal,
  Card,
} from "@nextui-org/react";
import { useRouter } from "next/router";
import Head from "next/head";
import styles from "@/styles/styles.module.css";

function Nav() {
  const router = useRouter();

  const handler = () => {
    router.push("/logout");
  };

  return (
    <>
      <Head>
        <link rel="shortcut icon" href="/Lyre.png" />
      </Head>
      <div className={styles.navbarStyle}>
        <Link href="/" style={{ marginRight: "15px" }}>
          <Image
            width={188}
            height={75}
            src="/Lyre.png"
            alt="Logo"
            style={{ objectFit: "contain" }}
          />
        </Link>
        <Link
          className={styles.linkStyle}
          style={{ fontSize: "30px" }}
          href="/"
        >
          Orpheus
        </Link>
        <Link
          className={styles.linkStyle}
          style={{ fontSize: "30px" }}
          href="/profile"
        >
          Profile
        </Link>
        <Link
          className={styles.linkStyle}
          style={{ fontSize: "30px" }}
          href="/about"
        >
          About
        </Link>
        <div className={styles.logoutContainerStyle}>
          <Button
            className={styles.buttonStyle}
            style={{ background: "red" }}
            onClick={handler}
          >
            Logout
          </Button>
        </div>
      </div>
    </>
  );
}
function NavNotLogged() {
  return (
    <>
      <Head>
        <link rel="shortcut icon" href="/Lyre.png" />
      </Head>
      <div className={styles.navbarStyle}>
        <Link href="/" style={{ marginRight: "15px" }}>
          <Image
            width={188}
            height={75}
            src="/Lyre.png"
            alt="Logo"
            style={{ objectFit: "contain" }}
          />
        </Link>
        <Link
          className={styles.linkStyle}
          style={{ fontSize: "30px" }}
          href="/about"
        >
          About
        </Link>
      </div>
    </>
  );
}
module.exports = {
  Nav,
  NavNotLogged,
};
