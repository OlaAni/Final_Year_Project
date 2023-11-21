"use client";

import Image from "next/image";
import styles from "./page.module.css";
import Orpheus from "./test";
import { NextUIProvider } from "@nextui-org/react";

export default function Home() {
  return (
    <NextUIProvider>
      <Orpheus />
    </NextUIProvider>
  );
}
