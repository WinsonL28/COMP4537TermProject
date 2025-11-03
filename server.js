const http = require('http');
const axios = require('axios');
require('dotenv').config();


const MODEL_SERVER_URL = `http://${process.env.MODEL_SERVER_IP}:5000/generate`;

const server = http.createServer((req, res) => {

    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    // Handle preflight requests
    if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
    }

    if (req.method === 'POST' && req.url === '/dnd') {
        let body = '';

        req.on('data', chunk => {
            body += chunk.toString();
        });

        req.on('end', async () => {
            try {
                const { prompt } = JSON.parse(body);
                const response = await axios.post(MODEL_SERVER_URL, { prompt });

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ result: response.data.response }));
            } catch (error) {
                console.error('Error contacting model server:', error.message);
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Model server unreachable' }));
            }
        });
    } else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Not found' }));
    }
});

server.listen(5000, () => {
    console.log('DD backend running on port 5000');
});