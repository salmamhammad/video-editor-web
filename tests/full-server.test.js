// server.test.js
const request = require("supertest");
const { app, server } = require("./server");

afterAll((done) => {
  server.close(done);
});

describe("Server basic routes", () => {
  it("should load login page", async () => {
    const res = await request(app).get("/login");
    expect(res.statusCode).toEqual(200);
    expect(res.text).toContain("<!DOCTYPE html>");
  });

  it("should load signup page", async () => {
    const res = await request(app).get("/signup");
    expect(res.statusCode).toEqual(200);
  });

  it("should reject root URL without token", async () => {
    const res = await request(app).get("/");
    expect(res.statusCode).toEqual(302); // redirected to login
  });

  it("should upload a video file", async () => {
    const res = await request(app)
      .post("/upload")
      .attach("video", Buffer.from("dummycontent"), "test.mp4");
    expect(res.statusCode).toEqual(200);
    expect(res.body).toHaveProperty("filename");
  });
});
