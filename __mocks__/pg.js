const mockQuery = jest.fn();

const Pool = jest.fn(() => ({
  query: mockQuery
}));

Pool.prototype.query = mockQuery;

mockQuery.mockImplementation((text, params) => {
  if (text.includes('SELECT * FROM users WHERE email =')) {
    if (params[0] === 'nonexistent@test.com') {
      return Promise.resolve({ rows: [] });
    }
    return Promise.resolve({
      rows: [{ id: 1, name: 'Test User', email: params[0], password_hash: '$2a$10$mockedhash' }]
    });
  }

  if (text.includes('INSERT INTO users')) {
    return Promise.resolve();
  }

  return Promise.resolve({ rows: [] });
});

module.exports = { Pool };
