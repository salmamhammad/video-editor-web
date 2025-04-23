const request = require('supertest');
const app = require('../server'); // now it's in the same directory

describe('GET /', () => {
  it('should return 200 OK', async () => {
    const res = await request(app).get('/');
    expect(res.statusCode).toBe(200);
    expect(res.text).toContain("Hello");
  });
});
