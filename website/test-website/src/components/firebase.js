import { initializeApp } from "firebase/app";
import { get, getDatabase } from "firebase/database";
import { getStorage } from "firebase/storage";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyC-FgcIra0uGsn5Aq6aaXbZ-KHTCGKe3zI",
  authDomain: "orpheus-3a4fa.firebaseapp.com",
  databaseURL:
    "https://orpheus-3a4fa-default-rtdb.europe-west1.firebasedatabase.app",
  projectId: "orpheus-3a4fa",
  storageBucket: "orpheus-3a4fa.appspot.com",
  messagingSenderId: "972087076700",
  appId: "1:972087076700:web:30b79a4fc7d6a4d3cf0399",
  measurementId: "G-9X42R7LWF7",
};

const app = initializeApp(firebaseConfig);
const database = getDatabase(app);
const storage = getStorage(app);
const auth = getAuth(app);

module.exports = {
  firebaseConfig,
  auth,
  database,
  storage,
};
