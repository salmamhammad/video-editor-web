const request = require('supertest');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const fs = require('fs');
const { app, server } = require('../server'); // Ensure app is exported

jest.mock('jsonwebtoken');
jest.mock('bcryptjs');
jest.mock('fs');
jest.mock('child_process', () => ({
  exec: jest.fn((cmd, cb) => cb(null, 'mocked stdout', ''))
}));

describe('Express Server', () => {
  afterAll(() => {
    if (server && server.close) server.close();
  });

  describe('JWT Handling', () => {
    it('should decode and verify JWT token', async () => {
      jwt.verify.mockImplementation(() => ({ user: 'testuser' }));

      const res = await request(app)
        .get('/your-protected-route') // Replace with your actual route
        .set('Authorization', 'Bearer mocktoken');

      // expect actual status based on your app
      expect([200, 302, 403]).toContain(res.statusCode);
    });

    it('should reject invalid JWT token', async () => {
      jwt.verify.mockImplementation(() => {
        throw new Error('Invalid token');
      });

      const res = await request(app)
        .get('/your-protected-route') // Replace with your actual route
        .set('Authorization', 'Bearer invalidtoken');

      expect([401, 403]).toContain(res.statusCode);
    });
  });

  describe('File Upload', () => {
    it('should upload a file successfully', async () => {
      const res = await request(app)
        .post('/process') // ← change this to your real upload endpoint
        .attach('video', Buffer.from('dummy content'), 'video.mp4');

      expect([200, 201]).toContain(res.statusCode);
    });

    it('should return error without file', async () => {
      const res = await request(app)
        .post('/process'); // ← same endpoint

      expect([400, 422]).toContain(res.statusCode);
    });
  });
});
