const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Store active processes
let pythonProcess = null;

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'views', 'index.html'));
});

app.post('/api/generate', async (req, res) => {
  const { prompt, max_tokens = 150, temperature = 0.7 } = req.body;
  
  if (!prompt) {
    return res.status(400).json({ error: 'Prompt is required' });
  }

  try {
    const response = await generateWithMistral(prompt, max_tokens, temperature);
    res.json(response);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    model: 'Mistral 7B',
    memory: process.memoryUsage() 
  });
});

// Python process management
function startPythonProcess() {
  if (pythonProcess) {
    pythonProcess.kill();
  }

  pythonProcess = spawn('python3', ['model-handler.py']);
  
  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python Error: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    pythonProcess = null;
  });
}

function generateWithMistral(prompt, max_tokens, temperature) {
  return new Promise((resolve, reject) => {
    if (!pythonProcess) {
      startPythonProcess();
    }

    const request = {
      prompt: prompt,
      max_tokens: max_tokens,
      temperature: temperature
    };

    pythonProcess.stdin.write(JSON.stringify(request) + '\n');
    
    const timeout = setTimeout(() => {
      reject(new Error('Generation timeout'));
    }, 30000);

    pythonProcess.stdout.once('data', (data) => {
      clearTimeout(timeout);
      try {
        const response = JSON.parse(data.toString());
        resolve(response);
      } catch (error) {
        reject(error);
      }
    });
  });
}

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
  console.log('🤖 Mistral 7B API ready');
  startPythonProcess();
});

// Cleanup on exit
process.on('SIGINT', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  process.exit();
});