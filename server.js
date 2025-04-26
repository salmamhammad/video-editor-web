// server.js
const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const { exec } = require("child_process");
const WebSocket = require("ws");
const jwt = require("jsonwebtoken");
const bcrypt = require("bcryptjs");
const http = require("http");
const { Pool } = require("pg");

// Constants
const PORT = 8082;
const SECRET = 'secret';

// Express app setup
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });
const pool = new Pool({
  host: "db",
  user: "admin",
  password: "password",
  database: "video_editor",
  port: 5432,
});

app.use(cors({ origin: "*" }));
app.use(express.json());
app.use("/backend/uploads", express.static(path.join(__dirname, "backend/uploads")));
app.use("/processed", express.static(path.join(__dirname, "processed")));
app.use("/videos", express.static(path.join(__dirname, "backend/uploads")));
app.use(express.static(path.join(__dirname, 'public')));

// WebSocket setup
const clients = new Map();
let pythonSocket = null;

// WebSocket server: client side
wss.on("connection", (ws) => {
  const clientId = generateClientId();
  clients.set(clientId, ws);
  console.log(`Client connected: ${clientId}`);

  ws.on("message", (message) => {
    console.log(`Received from client ${clientId}:`, message.toString());
    if (pythonSocket && pythonSocket.readyState === WebSocket.OPEN) {
      pythonSocket.send(JSON.stringify({ clientId, message: message.toString() }));
    }
  });

  ws.on("close", () => {
    console.log(`Client disconnected: ${clientId}`);
    clients.delete(clientId);
  });

  ws.on("error", (error) => {
    console.error(`WebSocket error for client ${clientId}:`, error);
  });
});

// Authentication middleware
function verifyToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  if (!token) return res.redirect('/login.html');

  jwt.verify(token, SECRET, (err, user) => {
    if (err) return res.redirect('/login.html');
    req.user = user;
    next();
  });
}

// Multer setup for file uploads
const storage = multer.diskStorage({
  destination: "./backend/uploads/",
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});
const upload = multer({ storage });

// Routes
app.get('/', verifyToken, (req, res) => {
  if (!req.headers['authorization']) {
    return res.redirect(302, '/login'); // Ensure this is a redirect (302)
  }
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/login', (req, res) => res.sendFile(path.join(__dirname, 'public', 'login.html')));
app.get('/signup', (req, res) => res.sendFile(path.join(__dirname, 'public', 'signup.html')));
app.get('/editor', (req, res) => res.sendFile(path.join(__dirname, 'public', 'editor.html')));

// Upload single file
app.post("/uploaddata", upload.single("file"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });
  res.json({ fileUrl: req.file.filename });
});

// Upload video
app.post("/upload", upload.single("video"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });
  res.json({
    message: "Video uploaded successfully",
    filename: req.file.filename,
    downloadUrl: `http://localhost:${PORT}/videos/${req.file.filename}`,
  });
});

// Video Processing APIs
function processVideo(script, args, res) {
  exec(`python ${script} ${args.join(' ')}`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Processing error: ${stderr}`);
      return res.status(500).json({ error: "Processing failed" });
    }
    res.json({ message: "Processing completed", outputFile: args[args.length - 1], nameFile: path.basename(args[args.length - 1]) });
  });
}

app.post("/process", upload.single("video"), (req, res) => {
  const random = randomFilename("video", "mp4");
  processVideo("process.py", [
    `backend/uploads/${req.body.file}`,
    req.body.src,
    req.body.clusters,
    `backend/uploads/${random}`
  ], res);
});

app.post("/denoise", upload.single("video"), (req, res) => {
  const random = randomFilename("audio", "wav");
  processVideo("denoise.py", [
    `backend/uploads/${req.body.file}`,
    `backend/uploads/${random}`
  ], res);
});

app.post("/blur", upload.single("video"), (req, res) => {
  const random = randomFilename("video", "mp4");
  processVideo("ffmpeg1.py", [
    "blur",
    `backend/uploads/${req.body.file}`,
    `backend/uploads/${random}`
  ], res);
});

app.post("/color", upload.single("video"), (req, res) => {
  const random = randomFilename("video", "mp4");
  processVideo("ffmpeg1.py", [
    "color",
    `backend/uploads/${req.body.file}`,
    `backend/uploads/${random}`
  ], res);
});

// Auth: signup, login
app.post("/api/signup", async (req, res) => {
  try {
    const { name, email, password } = req.body;
    const hashed = await bcrypt.hash(password, 10);

    const existing = await pool.query("SELECT * FROM users WHERE email = $1", [email]);
    if (existing.rows.length > 0) {
      return res.status(400).send({ success: false, error: "Email already registered" });
    }

    const newUser = await pool.query(
      "INSERT INTO users (name, email, password_hash) VALUES ($1, $2, $3) RETURNING *",
      [name, email, hashed]
    );
    const token = jwt.sign({ id: newUser.rows[0].id }, SECRET);
    res.status(201).send({ success: true, token, name });
  } catch (err) {
    console.error("Signup Error:", err);
    res.status(500).send({ success: false, error: "Signup failed" });
  }
});

app.post("/api/login", async (req, res) => {
  const { email, password } = req.body;
  const result = await pool.query("SELECT * FROM users WHERE email = $1", [email]);
  const user = result.rows[0];
  if (!user || !(await bcrypt.compare(password, user.password_hash))) {
    return res.status(401).send({ error: "Invalid credentials" });
  }
  const token = jwt.sign({ id: user.id }, SECRET);
  res.send({ token, name: user.name });
});

// Projects
app.post("/api/projects", async (req, res) => {
  const { userId, name, data } = req.body;
  await pool.query("INSERT INTO projects (user_id, name, data) VALUES ($1, $2, $3)", [userId, name, data]);
  res.send({ success: true });
});

app.get("/api/projects/:userId", async (req, res) => {
  const { userId } = req.params;
  const result = await pool.query("SELECT * FROM projects WHERE user_id = $1", [userId]);
  res.send(result.rows);
});

// Connect to Python WebSocket server
pythonSocket = new WebSocket("ws://backend:8765");

pythonSocket.on("open", () => console.log("Connected to Python WebSocket"));
pythonSocket.on("message", (message) => {
  try {
    const data = JSON.parse(message.toString());
    const client = clients.get(data.clientId);
    if (client) {
      client.send(JSON.stringify({ clientId: data.clientId, response: data.response }));
      console.log(`Sent response to client ${data.clientId}`);
    }
  } catch (err) {
    console.error("Error handling message from Python:", err);
  }
});
pythonSocket.on("close", () => console.log("Python WebSocket closed"));
pythonSocket.on("error", (err) => console.error("Python WebSocket error:", err));

// Helpers
function generateClientId() {
  return `client-${Math.random().toString(36).substr(2, 9)}`;
}

function randomFilename(prefix, ext) {
  const random = Math.floor(Math.random() * 900000) + 100000;
  return `${prefix}_${random}.${ext}`;
}

// Start server
// server.listen(PORT, () => {
//   console.log(`🚀 Server running at http://localhost:${PORT}`);
// });

// Export for tests
module.exports = { app, server };
