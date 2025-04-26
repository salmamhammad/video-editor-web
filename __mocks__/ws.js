module.exports = function WebSocket() {
    this.on = jest.fn();
    this.send = jest.fn();
    this.close = jest.fn();
  };
  module.exports.Server = jest.fn(() => ({
    on: jest.fn(),
  }));