import { createUserWithEmailAndPassword } from "firebase/auth";

import { auth } from "@/components/firebase";

export default async function handler(req, res) {
  var password = req.body.password;
  var email = req.body.email;

  createUserWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      const user = userCredential.user;
      res.status(200).json({
        status: "OK",
      });
    })
    .catch((error) => {
      const errorCode = error.code;
      const errorMessage = error.message;
      res.status(400).json({
        status: "Bad",
      });
      return;
    });
}
