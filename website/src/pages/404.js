import * as React from "react";

const { Nav } = require("@/components/Nav");
import Head from "next/head";
import styles from "@/styles/styles.module.css";

import { NextUIProvider, Text } from "@nextui-org/react";
export default function Logout() {
  return (
    <NextUIProvider>
      <Nav />
      <Head>
        <title>NOT FOUND</title>
      </Head>
      <Text className={styles.sectionTitleStyle}>404...Page Not Found</Text>
    </NextUIProvider>
  );
}
