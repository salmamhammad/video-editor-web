const mockQuery = jest.fn();
exports.Pool = jest.fn(() => ({
  query: mockQuery,
}));