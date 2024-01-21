import { Button, Link, Table, TableColumn, Spacer } from "@nextui-org/react";
import { useRouter } from "next/router";

function Nav() {
  const router = useRouter();

  const handler = () => {
    router.push("/logout");
  };

  return (
    <>
      <div
        style={{ display: "flex", justifyContent: "center", marginTop: "20px" }}
      >
        <Link href="/">Orpheus</Link>
        <Spacer y={3} />
        <Link href="/profile">Profile</Link>
      </div>
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <Button onPress={handler}>Logout</Button>
      </div>
    </>
  );
}

module.exports = {
  Nav,
};
