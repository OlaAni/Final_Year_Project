import Image from "next/image";
import styles from "./page.module.css";
import Orpheus from "./test";

export default function Home() {
  return (
    <main className={styles.main}>
      <Orpheus />
    </main>
  );
}
