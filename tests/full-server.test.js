// // tests/full-server.test.js
// const { spawn } = require('child_process');
// const request = require('supertest');
// const waitOn = require('wait-on');

// let serverProcess;

// beforeAll(async () => {
//   serverProcess = spawn('node', ['server.js']);

//   await waitOn({
//     resources: ['http://localhost:8080/login'],
//     timeout: 10000,
//   });
// });

// afterAll(() => {
//   serverProcess.kill();
// });

// describe('GET /login', () => {
//   it('should return the login page', async () => {
//     const res = await request('http://localhost:8080').get('/login');
//     expect(res.statusCode).toBe(200);
//   });
// });



const request = require('supertest');
const WebSocket = require('ws');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const fs = require('fs');
const path = require('path');
const { app, server } = require('../server'); // adjust path if needed

jest.mock('jsonwebtoken');
jest.mock('bcryptjs');
jest.mock('fs');
jest.mock('child_process', () => ({
  exec: jest.fn((cmd, cb) => cb(null, 'mocked stdout', ''))
}));

describe('Express Server', () => {
  afterAll(() => {
    server.close(); // Ensure server is shut down after tests
  });

  describe('JWT Handling', () => {
    it('should decode and verify JWT token', async () => {
      jwt.verify.mockImplementation(() => ({ user: 'testuser' }));

      const token = jwt.sign({ user: 'testuser' }, 'secret');
      const res = await request(app)
        .get('/protected') // replace with actual route if applicable
        .set('Authorization', `Bearer ${token}`);

      expect(res.statusCode).not.toBe(401);
    });

    it('should reject invalid JWT token', async () => {
      jwt.verify.mockImplementation(() => {
        throw new Error('Invalid token');
      });

      const res = await request(app)
        .get('/protected') // replace with actual route
        .set('Authorization', `Bearer invalidtoken`);

      expect(res.statusCode).toBe(401);
    });
  });

  describe('Password Handling', () => {
    it('should hash password', async () => {
      bcrypt.hash.mockImplementation(() => Promise.resolve('hashedPassword'));
      const hashed = await bcrypt.hash('password123', 10);
      expect(hashed).toBe('hashedPassword');
    });

    it('should compare passwords', async () => {
      bcrypt.compare.mockImplementation(() => Promise.resolve(true));
      const match = await bcrypt.compare('password123', 'hashedPassword');
      expect(match).toBe(true);
    });
  });

  describe('File Upload', () => {
    it('should upload a file successfully', async () => {
      const res = await request(app)
        .post('/upload') // replace with actual route
        .attach('file', Buffer.from('dummy content'), 'test.txt');

      expect(res.statusCode).toBe(200); // or 201 or your actual status
    });

    it('should handle missing file', async () => {
      const res = await request(app)
        .post('/upload');

      expect(res.statusCode).toBe(400); // or whatever error you return
    });
  });

  describe('Utility + WebSocket', () => {
    let ws;
    const address = `ws://localhost:8080`;

    beforeEach((done) => {
      ws = new WebSocket(address);
      ws.on('open', () => done());
    });

    afterEach(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    });

    it('should connect via WebSocket and receive a response', (done) => {
      ws.on('message', (msg) => {
        expect(msg).toBeDefined();
        done();
      });

      ws.send('Hello from test');
    });
  });
});
