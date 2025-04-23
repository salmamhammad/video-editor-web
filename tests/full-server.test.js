// tests/full-server.test.js
const { spawn } = require('child_process');
const request = require('supertest');
const waitOn = require('wait-on');

let serverProcess;

beforeAll(async () => {
  serverProcess = spawn('node', ['server.js']);

  await waitOn({
    resources: ['http://localhost:8080/login'],
    timeout: 10000,
  });
});

afterAll(() => {
  serverProcess.kill();
});

describe('GET /login', () => {
  it('should return the login page', async () => {
    const res = await request('http://localhost:8080').get('/login');
    expect(res.statusCode).toBe(200);
  });
});
