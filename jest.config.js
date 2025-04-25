module.exports = {
    testEnvironment: 'node',
    setupFiles: ['<rootDir>/jest.setup.js'],
    moduleNameMapper: {
      '^pg$': '<rootDir>/__mocks__/pg.js' // Redirect pg import to mock
    }
  };
  