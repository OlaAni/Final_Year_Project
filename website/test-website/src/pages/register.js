import { useRouter } from "next/router";

import * as React from "react";

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
      router.push("/");
    } else {
      alert(result.status);
    }
  }

  return (
    <NextUIProvider>
      <Spacer y={1} />
      <Spacer y={2.5} />
      <Row gap={1}>
        <Card css={{ $$cardColor: "lightGreen", mw: "600px", margin: "auto" }}>
          <Card.Body>
            <Text h6 align="center" size={36} css={{ m: 0 }}>
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
                    <Button color="secondary" size="lg" type="submit">
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
