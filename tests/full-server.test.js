jest.mock('pg');
jest.mock('ws');

process.env.NODE_ENV = 'test';
const request = require('supertest');
const path = require('path');

let app, server;



beforeAll(() => {
    const serverModule = require(path.resolve(__dirname, '../server.js'));
    app = serverModule.app || serverModule;
    server = serverModule.server;
  });
  
  afterAll((done) => {
    if (server && server.close) {
      server.close(done);
    } else {
      done();
    }
  });

describe('🚀 Server Tests', () => {
  test('GET /login should return 200 and serve login.html', async () => {
    const res = await request(app).get('/login');
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-type']).toContain('text/html');
  });

  test('POST /api/signup should create user or respond with error', async () => {
    const response = await request(app)
      .post('/api/signup')
      .send({
        name: 'Test User',
        email: `test${Date.now()}@test.com`,
        password: 'testpass123'
      })
      .set('Accept', 'application/json');
    
    expect(response.statusCode).toBeGreaterThanOrEqual(200);
    expect(response.statusCode).toBeLessThan(500);
    expect(response.body).toHaveProperty('success');
  });

  test('POST /api/login should reject invalid credentials', async () => {
    const response = await request(app)
      .post('/api/login')
      .send({
        email: 'nonexistent@test.com',
        password: 'wrongpass'
      });

    expect(response.statusCode).toBe(401);
    expect(response.body).toHaveProperty('error');
  });
});
