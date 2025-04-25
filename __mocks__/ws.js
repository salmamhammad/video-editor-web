module.exports = class WebSocket {
    constructor(url) {
      this.url = url;
      this.readyState = 1;
      setTimeout(() => {
        if (this.onopen) this.onopen();
      }, 10);
    }
    send() {}
    close() {}
    on() {}
  };
  module.exports.Server = class {
    constructor() {
      this.on = jest.fn();
    }
  };
  