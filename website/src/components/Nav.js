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

function Nav() {
  const router = useRouter();

  const handler = () => {
    router.push("/logout");
  };
  const navbarStyle = {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "10px 20px",
    background: "#daa520",
  };

  const linkStyle = {
    color: "#black",
    textDecoration: "none",
    margin: "0 15px",
    fontSize: "18px",
  };

  const logoutContainerStyle = {
    display: "flex",
    justifyContent: "flex-end",
    alignItems: "center",
  };

  const buttonStyle = {
    background: "red",
    color: "#fff",
    padding: "8px 12px",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
  };
  return (
    <>
      <div style={navbarStyle}>
        <Link href="/" style={{ marginRight: "15px" }}>
          <Image
            width={188}
            height={75}
            src="/Lyre.png"
            alt="Logo"
            style={{ objectFit: "contain" }}
          />
        </Link>
        <Link style={linkStyle} href="/">
          Orpheus
        </Link>
        <Link style={linkStyle} href="/profile">
          Profile
        </Link>
        <div style={logoutContainerStyle}>
          <button style={buttonStyle} onClick={handler}>
            Logout
          </button>
        </div>
      </div>
    </>
  );
}

module.exports = {
  Nav,
};
