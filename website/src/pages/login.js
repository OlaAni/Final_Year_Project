import { useRouter } from "next/router";
import {
  NextUIProvider,
  Container,
  Card,
  Row,
  Text,
  Col,
  Spacer,
  Button,
  Input,
  Image,
  Link,
  Grid,
} from "@nextui-org/react";
import Head from "next/head";

export default function Login() {
  const router = useRouter();

  async function handleSubmit(event) {
    event.preventDefault();

    const data = {
      email: event.target.email.value,
      password: event.target.password.value,
    };

    const options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    };

    const response = await fetch("api/profile/login", options);
    const result = await response.json();
    if (result.status == "OK") {
      router.push("/");
    } else {
      alert("Status: " + result.status);
    }
  }

  return (
    <NextUIProvider>
      <Head>
        <title>Login</title>
        <link rel="shortcut icon" href="/Lyre.png" />
      </Head>
      <Spacer y={2} />
      <Row gap={1}>
        <Card css={{ $$cardColor: "#daa520", mw: "600px", margin: "auto" }}>
          <Card.Body>
            <Text h6 align="center" size={40} css={{ m: 0 }}>
              Login
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
                />
              </Row>

              <Spacer y={2} />

              <Row>
                <Spacer />
                <Input.Password
                  size="xl"
                  fullWidth
                  id="password"
                  clearable
                  labelPlaceholder="Password"
                  minLength={3}
                />
              </Row>

              <Spacer y={2.5} />

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
                      Sign In
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
                        href="/register"
                      >
                        Register
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
