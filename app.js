// app.js
const express = require('express');
const app = express();
const port = 3000; // A common port for Node.js apps

// Define a route for the homepage
app.get('/', (req, res) => {
  res.send('Hello, World! Our JavaScript server is running!');
});

// Start the server
app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});