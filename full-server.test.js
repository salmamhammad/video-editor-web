jest.mock('jsonwebtoken', () => ({
    verify: jest.fn(() => ({ user: 'testuser' })),
  }));
  
  const request = require('supertest');
  const jwt = require('jsonwebtoken');
  const fs = require('fs');
  const { app, server } = require('server');
  
  let serverInstance;
  
  beforeAll((done) => {
    serverInstance = server.listen(0, () => {
      console.log('✅ Test server started');
      done();
    });
  });
  
  // Mocks
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
    afterAll(() => {
      return new Promise((resolve, reject) => {
        if (serverInstance && serverInstance.close) {
          serverInstance.close((err) => {
            if (err) reject(err);
            else {
              console.log('🛑 Test server stopped');
              resolve();
            }
          });
        } else {
          resolve();
        }
      });
    });
  
    describe('JWT Handling', () => {
      it('should decode and verify JWT token', async () => {
        const res = await request(app)
          .get('/protected')
          .set('Authorization', 'Bearer fakeToken');
  
        expect(res.statusCode).toBe(200);
        expect(res.body.user).toEqual('testuser');
        expect(jwt.verify).toHaveBeenCalledWith('fakeToken', 'secret');
      }, 10000);
  
      it('should reject if no token provided', async () => {
        const res = await request(app).get('/protected');
        expect(res.statusCode).toBe(401);
      }, 10000);
  
      it('should return 403 if token is invalid', async () => {
        jwt.verify.mockImplementation(() => {
          throw new Error('Invalid token');
        });
  
        const res = await request(app)
          .get('/protected')
          .set('Authorization', 'Bearer invalid');
  
        expect(res.statusCode).toBe(403);
      }, 10000);
    });
  
    describe('File Upload', () => {
      it('should upload a file successfully', async () => {
        const res = await request(serverInstance)
          .post('/process')
          .attach('video', Buffer.from('dummy content'), {
            filename: 'test.mp4',
            contentType: 'video/mp4'
          });
  
        expect([200, 201]).toContain(res.statusCode);
      }, 10000);
  
      it('should return error without file', async () => {
        const res = await request(serverInstance).post('/process');
        expect([400, 422]).toContain(res.statusCode);
      }, 10000);
    });
  });
  