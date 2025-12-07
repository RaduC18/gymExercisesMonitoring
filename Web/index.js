import express from "express";
import {dirname} from "path";
import { fileURLToPath} from "url";

const app = express();
const port = 3000;

const __dirname = dirname(fileURLToPath(import.meta.url));

app.use(express.static("public"));

app.get("/", (req, res) => {
  res.sendFile(__dirname + "/views/paginaPrincipala.html");
});


app.get("/Cardiac", (req, res) => {
  res.sendFile(__dirname + "/views/aritmie.html");
});

app.get("/Exercitii", (req, res) =>{
  res.sendFile(__dirname + "/views/exercitii.html")
});

app.get("/paginaPrincipala", (req, res) =>{
  res.sendFile(__dirname + "/views/paginaPrincipala.html");
});


app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});


