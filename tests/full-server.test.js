const request = require('supertest');
const jwt = require('jsonwebtoken');
const fs = require('fs');
const { app } = require('../server');
let serverInstance;
beforeAll((done) => {
    serverInstance = server.listen(0, () => {
      console.log('✅ Test server started');
      done();
    });
  });
// Mocks
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

describe('Express Server', () => {
  afterAll((done) => {
    if (server && server.close) {
      server.close(done);
    } else {
      done();
    }
  });

  describe('JWT Handling', () => {
    it('should decode and verify JWT token', async () => {
      const res = await request(app)
        .get('/protected')
        .set('Authorization', 'Bearer mocktoken');
  
      expect(res.statusCode).toBe(200);
      expect(res.body.user).toEqual('testuser');
    });
  });

  describe('File Upload', () => {
    it('should upload a file successfully', async () => {
      const res = await request(serverInstance)
        .post('/process') // ✅ Your actual upload route
        .attach('video', Buffer.from('dummy'), {
          filename: 'test.mp4',
          contentType: 'video/mp4'
        });

      expect([200, 201]).toContain(res.statusCode);
    }, 10000);

    it('should return error without file', async () => {
      const res = await request(serverInstance)
        .post('/process');

      expect([400, 422]).toContain(res.statusCode);
    });
  });
});
afterAll((done) => {
    serverInstance.close(() => {
      console.log('🛑 Test server stopped');
      done();
    });
  });