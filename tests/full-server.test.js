// tests/server.test.js

const request = require('supertest');
const http = require('http');
const app = require('../server.js'); // <== Make sure you export `app` from server.js
const { exec } = require('child_process');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { Pool } = require('pg');

// Mocks
jest.mock('child_process');
jest.mock('pg');
jest.mock('bcryptjs');
jest.mock('jsonwebtoken');

let server;

beforeAll((done) => {
  server = http.createServer(app);
  server.listen(0, done); // Random free port
});

afterAll((done) => {
  server.close(done);
});

// Mock DB connection
const mockQuery = jest.fn();
Pool.mockImplementation(() => ({
  query: mockQuery,
}));

describe('Public routes', () => {
  it('should load login page', async () => {
    const res = await request(server).get('/login');
    expect(res.statusCode).toBe(200);
  });

  it('should load signup page', async () => {
    const res = await request(server).get('/signup');
    expect(res.statusCode).toBe(200);
  });

  it('should load editor page', async () => {
    const res = await request(server).get('/editor');
    expect(res.statusCode).toBe(200);
  });
});

describe('Authentication and protected routes', () => {
  it('should redirect to login if no token', async () => {
    const res = await request(server).get('/');
    expect(res.statusCode).toBe(302);
    expect(res.headers.location).toBe('/login.html');
  });

  it('should allow access with valid token', async () => {
    jwt.verify.mockImplementation((token, secret, cb) => cb(null, { id: 1 }));
    const res = await request(server).get('/').set('Authorization', 'Bearer faketoken');
    expect(res.statusCode).toBe(200);
  });
});

describe('Uploads and Processing', () => {
  it('should upload data', async () => {
    const res = await request(server)
      .post('/uploaddata')
      .attach('file', Buffer.from('dummy file'), 'test.mp4');

    expect(res.statusCode).toBe(200);
    expect(res.body.fileUrl).toBeDefined();
  });

  it('should upload video', async () => {
    const res = await request(server)
      .post('/upload')
      .attach('video', Buffer.from('dummy video'), 'video.mp4');

    expect(res.statusCode).toBe(200);
    expect(res.body.filename).toBeDefined();
  });

  it('should process video', async () => {
    exec.mockImplementation((cmd, cb) => cb(null, 'done', ''));
    const res = await request(server)
      .post('/process')
      .field('file', 'video.mp4')
      .field('src', 'src')
      .field('clusters', '3');

    expect(res.statusCode).toBe(200);
    expect(res.body.outputFile).toBeDefined();
  });

  it('should denoise audio', async () => {
    exec.mockImplementation((cmd, cb) => cb(null, 'done', ''));
    const res = await request(server)
      .post('/denoise')
      .field('file', 'audio.mp4');

    expect(res.statusCode).toBe(200);
    expect(res.body.outputFile).toBeDefined();
  });

  it('should blur video', async () => {
    exec.mockImplementation((cmd, cb) => cb(null, 'done', ''));
    const res = await request(server)
      .post('/blur')
      .field('file', 'video.mp4');

    expect(res.statusCode).toBe(200);
    expect(res.body.outputFile).toBeDefined();
  });

  it('should color correct video', async () => {
    exec.mockImplementation((cmd, cb) => cb(null, 'done', ''));
    const res = await request(server)
      .post('/color')
      .field('file', 'video.mp4');

    expect(res.statusCode).toBe(200);
    expect(res.body.outputFile).toBeDefined();
  });
});

describe('User Authentication API', () => {
  it('should signup a user', async () => {
    bcrypt.hash.mockResolvedValue('hashedpassword');
    mockQuery.mockResolvedValueOnce({ rows: [] });
    mockQuery.mockResolvedValueOnce({ rows: [{ id: 1, name: 'Test User' }] });

    const res = await request(server)
      .post('/api/signup')
      .send({ name: 'Test', email: 'test@test.com', password: 'pass123' });

    expect(res.statusCode).toBe(201);
    expect(res.body.token).toBeDefined();
  });

  it('should login a user', async () => {
    bcrypt.compare.mockResolvedValue(true);
    mockQuery.mockResolvedValueOnce({
      rows: [{ id: 1, name: 'Test', password_hash: 'hashed' }],
    });

    const res = await request(server)
      .post('/api/login')
      .send({ email: 'test@test.com', password: 'pass123' });

    expect(res.statusCode).toBe(200);
    expect(res.body.token).toBeDefined();
  });

  it('should fail login with wrong credentials', async () => {
    bcrypt.compare.mockResolvedValue(false);
    mockQuery.mockResolvedValueOnce({ rows: [{ id: 1, password_hash: 'hashed' }] });

    const res = await request(server)
      .post('/api/login')
      .send({ email: 'test@test.com', password: 'wrongpassword' });

    expect(res.statusCode).toBe(401);
  });
});

describe('Projects API', () => {
  it('should save a project', async () => {
    mockQuery.mockResolvedValueOnce({});

    const res = await request(server)
      .post('/api/projects')
      .send({ userId: 1, name: 'Project 1', data: {} });

    expect(res.statusCode).toBe(200);
    expect(res.body.success).toBe(true);
  });

  it('should load projects', async () => {
    mockQuery.mockResolvedValueOnce({ rows: [{ id: 1, name: 'Project 1' }] });

    const res = await request(server)
      .get('/api/projects/1');

    expect(res.statusCode).toBe(200);
    expect(res.body.length).toBeGreaterThan(0);
  });
});
