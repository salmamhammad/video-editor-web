const mockPool = {
    query: jest.fn(async (sql, params) => {
      // Mock signup email lookup
      if (sql.includes('SELECT * FROM users WHERE email =')) {
        return { rows: [] }; // No user found
      }
      // Mock signup insert
      if (sql.includes('INSERT INTO users')) {
        return { rowCount: 1 };
      }
      // Mock user returned after insert
      return {
        rows: [{
          id: 1,
          email: params[0],
          name: 'Test User',
          password_hash: await require('bcryptjs').hash('testpass123', 10),
        }]
      };
    }),
  };
  
  module.exports = {
    Pool: jest.fn(() => mockPool),
  };
  