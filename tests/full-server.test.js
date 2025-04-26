// Mocks at the top
jest.mock('jsonwebtoken', () => ({
  verify: jest.fn(() => ({ user: 'testuser' })),
}));

jest.mock('fs');
jest.mock('child_process', () => ({
  exec: jest.fn((cmd, cb) => cb(null, 'mocked stdout', '')),
}));
jest.mock('ws', () => {
  const mWebSocket = jest.fn(() => ({
    on: jest.fn(),
    send: jest.fn(),
    close: jest.fn(),
    readyState: 1,
  }));

  mWebSocket.Server = jest.fn(() => ({
    on: jest.fn(),
    clients: new Set(),
  }));

  return mWebSocket;
});

// Imports
const request = require('supertest');
const jwt = require('jsonwebtoken');
const fs = require('fs');

// 🛠 IMPORTANT: Correct path to server.js
const { app, server } = require('../server'); // <-- adjust if your server.js is elsewhere

let serverInstance;

beforeAll((done) => {
  serverInstance = server.listen(0, () => {
    console.log('✅ Test server started');
    done();
  });
});

afterAll((done) => {
  if (serverInstance) {
    serverInstance.close(() => {
      console.log('🛑 Test server stopped');
      done();
    });
  } else {
    done();
  }
});

// 🧪 Tests
describe('Express Server', () => {
  describe('JWT Handling', () => {
    it('should decode and verify JWT token', async () => {
      const res = await request(app)
        .get('/protected')
        .set('Authorization', 'Bearer fakeToken');

      expect(res.statusCode).toBe(200);
      expect(res.body.user).toEqual('testuser');
      expect(jwt.verify).toHaveBeenCalledWith('fakeToken', 'secret');
    });

    it('should reject if no token provided', async () => {
      const res = await request(app).get('/protected');
      expect(res.statusCode).toBe(401);
    });

    it('should return 403 if token is invalid', async () => {
      jwt.verify.mockImplementation(() => {
        throw new Error('Invalid token');
      });

      const res = await request(app)
        .get('/protected')
        .set('Authorization', 'Bearer invalid');

      expect(res.statusCode).toBe(403);
    });
  });

  describe('File Upload', () => {
    it('should upload a file successfully', async () => {
      const res = await request(serverInstance)
        .post('/process')
        .attach('video', Buffer.from('dummy'), {
          filename: 'test.mp4',
          contentType: 'video/mp4',
        });

      expect([200, 201]).toContain(res.statusCode);
    }, 10000); // 👈 increase timeout

    it('should return error without file', async () => {
      const res = await request(serverInstance)
        .post('/process');

      expect([400, 422]).toContain(res.statusCode);
    });
  });
});
