require('dotenv').config();
const express = require('express');
const axios = require('axios');
const swaggerUi = require('swagger-ui-express');
const YAML = require('yamljs');
const cors = require('cors');

const swaggerDocument = YAML.load('./swagger.yaml');
const app = express();
const PORT = process.env.PORT || 5000;
const MODEL_SERVER_URL = `http://${process.env.MODEL_SERVER_IP}:5000/generate`;

// Middleware
app.use(express.json());
app.use(cors()); // handles preflight OPTIONS automatically

// POST
app.post('/dnd', async (req, res) => {
    try {
        const prompt = req.body;
        const response = await axios.post(MODEL_SERVER_URL, { prompt });
        res.json({ result: response.data.response });
    } catch (error) {
        console.error('Error contacting model server:', error.message);
        res.status(500).json({ error: 'Model server unreachable' });
    }
});

// Swagger UI
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument,{
        swaggerOptions: {
            url: "model/api-docs/swagger.yaml"
        }
    })
);

// 404 handler
app.use((req, res) => {
    res.status(404).json({ error: 'Not found' });
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Swagger UI: http://localhost:${PORT}/api-docs`);
});
