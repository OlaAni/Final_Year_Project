import { useRouter } from "next/router";

import * as React from "react";
import Head from "next/head";

import {
  NextUIProvider,
  Card,
  Row,
  Text,
  Col,
  Spacer,
  Input,
  Button,
  Link,
  Grid,
  Image,
} from "@nextui-org/react";
const { NavNotLogged } = require("@/components/Nav");

export default function Register() {
  const router = useRouter();
  async function handleSubmit(event) {
    event.preventDefault();

    console.log("hello");
    const data = JSON.stringify({
      password: event.target.password.value,
      email: event.target.email.value,
    });

    const endpoint = "../api/profile/register";
    const options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: data,
    };
    const response = await fetch(endpoint, options);
    const result = await response.json();
    if (result.status == "OK") {
      router.push("/login");
    } else {
      alert(result.status);
    }
  }

  return (
    <NextUIProvider>
      <Head>
        <title>Register</title>
        <link rel="shortcut icon" href="/Lyre.png" />
      </Head>
      <NavNotLogged />
      <Spacer y={2} />
      <Row gap={1}>
        <Card css={{ $$cardColor: "#daa520", mw: "600px", margin: "auto" }}>
          <Card.Body>
            <Text h6 align="center" size={40} css={{ m: 0 }}>
              Register
            </Text>
            <Spacer y={2.5} />
            <form onSubmit={handleSubmit}>
              <Row>
                <Spacer />
                <Input
                  size="xl"
                  fullWidth
                  id="email"
                  clearable
                  labelPlaceholder="Email"
                  minLength={3}
                  maxLength={255}
                />
              </Row>
              <Spacer y={2} />
              <Row>
                <Spacer />
                <Input
                  size="xl"
                  fullWidth
                  id="password"
                  clearable
                  type={"password"}
                  labelPlaceholder="Password"
                  minLength={3}
                />
              </Row>

              <Spacer y={2} />

              <Grid.Container justify="center">
                <Grid>
                  <Col>
                    <Button
                      css={{
                        color: "#daa520",
                        backgroundColor: "black",
                      }}
                      type="submit"
                    >
                      Register
                    </Button>
                    <Spacer />
                    <Text align="center" size={20}>
                      or{" "}
                      <Link
                        css={{
                          marginLeft: "auto",
                          marginRight: "auto",
                          fontWeight: "bold",
                        }}
                        href="/login"
                      >
                        Login
                      </Link>
                    </Text>
                  </Col>
                </Grid>
              </Grid.Container>
            </form>
          </Card.Body>
        </Card>
      </Row>
    </NextUIProvider>
  );
}
