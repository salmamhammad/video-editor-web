const request = require('supertest');
const jwt = require('jsonwebtoken');
const fs = require('fs');
const { app, server } = require('../server');

// Mocks
jest.mock('jsonwebtoken');
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
      jwt.verify.mockReturnValue({ user: 'testuser' });

      const res = await request(app)
        .get('/protected') // ✅ Make sure this route exists in server.js
        .set('Authorization', 'Bearer validtoken');

      expect(res.statusCode).toBe(200);
      expect(res.body).toEqual({ user: 'testuser' });
    }, 10000);

    it('should reject invalid JWT token', async () => {
      jwt.verify.mockImplementation(() => {
        throw new Error('Invalid token');
      });

      const res = await request(app)
        .get('/protected')
        .set('Authorization', 'Bearer invalidtoken');

      expect([401, 403]).toContain(res.statusCode);
    }, 10000);
  });

  describe('File Upload', () => {
    it('should upload a file successfully', async () => {
      const res = await request(app)
        .post('/process') // ✅ Your actual upload route
        .attach('video', Buffer.from('dummy content'), {
          filename: 'test.mp4',
          contentType: 'video/mp4'
        });

      expect([200, 201]).toContain(res.statusCode);
    }, 10000);

    it('should return error without file', async () => {
      const res = await request(app)
        .post('/process');

      expect([400, 422]).toContain(res.statusCode);
    });
  });
});
