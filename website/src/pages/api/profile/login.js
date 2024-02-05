import { signInWithEmailAndPassword } from "firebase/auth";
import { getIronSession } from "iron-session";
import { auth } from "@/components/firebase";

export default async function handler(req, res) {
  var password = req.body.password;
  var email = req.body.email;

  signInWithEmailAndPassword(auth, email, password)
    .then(async (userCredential) => {
      // Signed in
      const user = userCredential.user;

      const session = await getIronSession(req, res, {
        cookieName: process.env.COOKIE_NAME,
        password: process.env.SESSION_PASSWORD,
      });

      session.email = user.email;
      session.userID = user.uid;
      await session.save();
      console.log(session.email);
      console.log(user.uid);

      res.status(200).json({
        status: "OK",
        email: email,
      });
    })
    .catch((error) => {
      const errorCode = error.code;
      const errorMessage = error.message;
      res.status(400).json({
        status: "Incorrect Username or Password",
      });
    });
}
