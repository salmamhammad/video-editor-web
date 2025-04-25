// Silence WebSocket connection attempts in tests
global.WebSocket = class {
    constructor() {}
    on() {}
    send() {}
    close() {}
  };
  