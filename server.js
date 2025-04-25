const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const { exec } = require("child_process");
const WebSocket = require('ws');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');

const app = express();
const PORT = 8080;
const http = require('http');
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const SECRET = 'secret';
// Store WebSocket clients with unique IDs
const clients = new Map();

// Handle WebSocket connections
wss.on("connection", (ws) => {
  const clientId = generateClientId(); // Assign a unique ID to each client
  clients.set(clientId, ws);

  console.log(`Client connected: ${clientId}`);

  ws.on("message", (message) => {        
      console.log(`Received from client ${clientId}:`, message.toString());

      // Forward message to Python WebSocket Server with client ID
      if (pythonSocket && pythonSocket.readyState === WebSocket.OPEN) {
          pythonSocket.send(JSON.stringify({ clientId, message: message.toString() }));
      }
  });

  ws.on("close", () => {
      console.log(`Client disconnected: ${clientId}`);
      clients.delete(clientId); // Remove client on disconnect
  });
  ws.on("error", (error) => {
    console.error(`🚨 WebSocket error for client ${clientId}:`, error);
});
});

// Middleware to parse Authorization header and verify token
function verifyToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.redirect('/login.html');
  }

  jwt.verify(token, SECRET, (err, user) => {
    if (err) return res.redirect('/login.html');
    req.user = user;
    next();
  });
}
// Generate a unique client ID
function generateClientId() {
  return `client-${Math.random().toString(36).substr(2, 9)}`;
}
const { Pool } = require("pg");
const pool = new Pool({
  host: "db",
  user: "admin",
  password: "password",
  database: "video_editor",
  port: 5433,
});
app.use(cors({ origin: "*" })); 
app.use(express.json());
// Serve files inside the 'processed' folder
app.use("/backend/uploads", express.static(path.join(__dirname, "backend/uploads")));
app.use("/processed", express.static(path.join(__dirname, "processed")));

// Serve static files from the 'public' folder
app.use(express.static(path.join(__dirname, 'public')));
// 🔐 Secure root route
app.get('/', verifyToken, (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Login screen
app.get('/login', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'login.html'));
});
app.get('/signup', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'signup.html'));
});
// editor screen
app.get('/editor', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'editor.html'));
});
// Set up storage for uploaded videos
const storage = multer.diskStorage({
  destination: "./backend/uploads/",
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname)); // Rename file to avoid conflicts
  },
});

const upload = multer({ storage: storage });

// Upload Route
app.post("/uploaddata", upload.single("file"), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
    }

    const fileUrl = req.file.filename;
    res.json({ fileUrl });
});

// API Endpoint to upload video
app.post("/upload", upload.single("video"), (req, res) => {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }
  
    // Provide the URL to download the uploaded video
    res.json({
      message: "Video uploaded successfully",
      filename: req.file.filename,
      downloadUrl: `http://localhost:8080/videos/${req.file.filename}`, // URL to download the video
    });
  });

// Serve uploaded files
app.use("/videos", express.static(path.join(__dirname, "backend/uploads")));

// server.listen(PORT, () => {
//   console.log(`🚀 WebSocket server is running on http://localhost:${PORT}`);
// });
if (require.main === module) {
  server.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}
app.post("/process", upload.single("video"), (req, res) => {
    console.log(req.body.file);
    const randomNum = Math.floor(Math.random() * 900000) + 100000;
    const fileName = `video_${randomNum}.mp4`;
    const inputFile = `backend/uploads/${req.body.file}`;
    const outputFile = `backend/uploads/${fileName}`;
  
    exec(`python process.py ${inputFile} ${req.body.src} ${req.body.clusters} ${outputFile}`, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error processing video: ${stderr}`);
        return res.status(500).json({ error: "Processing failed" });
      }
      res.json({ message: "Video processed",
        outputFile: outputFile ,
        nameFile:fileName
    });
    });
  });
  app.post("/denoise", upload.single("video"), (req, res) => {
    console.log(req.body.file);
    const randomNum = Math.floor(Math.random() * 900000) + 100000;
    const fileName = `audio_${randomNum}.wav`;
    const inputFile = `backend/uploads/${req.body.file}`;
    const outputFile = `backend/uploads/${fileName}`;
  
    exec(`python denoise.py ${inputFile} ${outputFile}`, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error processing video: ${stderr}`);
        return res.status(500).json({ error: "Processing failed" });
      }
      res.json({ message: "Video processed",
        outputFile: outputFile ,
        nameFile:fileName
    });
    });
  });
  app.post("/blur", upload.single("video"), (req, res) => {
   console.log(req.body.file);
   const randomNum = Math.floor(Math.random() * 900000) + 100000;
   const fileName = `video_${randomNum}.mp4`;
   const inputFile = `backend/uploads/${req.body.file}`;
   const outputFile = `backend/uploads/${fileName}`;
   console.log(outputFile);

    exec(`python ffmpeg1.py blur ${inputFile} ${outputFile}`, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error processing video: ${stderr}`);
        return res.status(500).json({ error: "Processing failed" });
      }
      res.json({ message: "Video processed",
        outputFile: outputFile ,
        nameFile:fileName
    });
    });
  });

  app.post("/color", upload.single("video"), (req, res) => {
    const randomNum = Math.floor(Math.random() * 900000) + 100000;
    const fileName = `video_${randomNum}.mp4`;
    const inputFile = `backend/uploads/${req.body.file}`;
    const outputFile = `backend/uploads/${fileName}`;
  
    exec(`python ffmpeg1.py color ${inputFile} ${outputFile}`, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error processing video: ${stderr}`);
        return res.status(500).json({ error: "Processing failed" });
      }
      res.json({ message: "Video processed",
        outputFile: outputFile ,
        nameFile:fileName
    });
    }); 
  });

 ///////////////////////////

// Signup
app.post("/api/signup", async (req, res) => {
  try {
    const { name,email, password } = req.body;
    const hashed = await bcrypt.hash(password, 10);
    const result = await pool.query("SELECT * FROM users WHERE email = $1", [email]);
   if(!result){
    await pool.query(
      "INSERT INTO users (name,email, password_hash) VALUES ($1, $2,$3)",
      [name,email, hashed]
    );
     result = await pool.query("SELECT * FROM users WHERE email = $1", [email]);

  }
    const user = result.rows[0];
    const token = jwt.sign({ id: user.id }, "secret");
    res.status(201).send({ success: true,token ,name });
  } catch (err) {
    console.error("Signup Error:", err);
    res.status(500).send({ success: false, error: "Signup failed" });
  }
});


// Login
app.post("/api/login", async (req, res) => {
  const { email, password } = req.body;
  const result = await pool.query("SELECT * FROM users WHERE email = $1", [email]);
  const user = result.rows[0];
  if (!user || !(await bcrypt.compare(password, user.password_hash))) {
    return res.status(401).send({ error: "Invalid credentials" });
  }
  const token = jwt.sign({ id: user.id }, "secret");
  const name= user.name; // Use ENV in production
  res.send({ token ,name});
});

// Save Project
app.post("/api/projects", async (req, res) => {
  const { userId, name, data } = req.body;
  await pool.query("INSERT INTO projects (user_id, name, data) VALUES ($1, $2, $3)", [userId, name, data]);
  res.send({ success: true });
});

// Load Projects
app.get("/api/projects/:userId", async (req, res) => {
  const { userId } = req.params;
  const result = await pool.query("SELECT * FROM projects WHERE user_id = $1", [userId]);
  res.send(result.rows);
});
//////////////////////////////////
 function blockSleep(ms) {
  const start = Date.now();
  while (Date.now() - start < ms) {}
}

// blockSleep(10000); // Blocks everything for 2 seconds
// console.log("After 10 seconds delay");
 // Start Python WebSocket Client
 
 pythonSocket = new WebSocket("ws://backend:8765");

 pythonSocket.on("open", () => {
     console.log("Connected to Python WebSocket");
 });

   pythonSocket.on("message", (message) => {
    try {
       const data = JSON.parse(message.toString());
       const clientId = data.clientId;
       const client = clients.get(clientId);

       if (client) {
        client.send(JSON.stringify({ clientId: data.clientId, response: data.response }));
        console.log(`Sent response to Client ${clientId}: ${data.response}`);
       } else {
           console.log(`Client ${clientId} not found`);
       }
      } catch (error) {
        console.error("❌ Error handling message from Python:", error);
    }
   });

 pythonSocket.on("close", () => {
     console.log("Python WebSocket Closed");
 });

 pythonSocket.on("error", (err) => {
     console.error("Python WebSocket Error:", err);
 });
 module.exports = { app, server };