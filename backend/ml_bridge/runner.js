import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Detect python executable (.venv if present, otherwise system python)
function getPythonCommand() {
  const venvPythonWin = path.resolve(__dirname, '../../.venv/Scripts/python.exe');
  if (fs.existsSync(venvPythonWin)) {
    return venvPythonWin;
  }
  return 'python';
}

/**
 * Executes the Python ML bridge locally to run inference and reasoning on an uploaded painting.
 * @param {string} imagePath - Absolute path to the painting image file.
 * @returns {Promise<object>} - Structured prediction and art-historical reasoning JSON.
 */
export function runInference(imagePath) {
  return new Promise((resolve, reject) => {
    const pythonExe = getPythonCommand();
    const scriptPath = path.resolve(__dirname, 'predict.py');

    console.log(`[Local ML Runner] Executing Python model on: ${path.basename(imagePath)}`);

    const pyProcess = spawn(pythonExe, [scriptPath, '--image', imagePath], {
      cwd: path.resolve(__dirname, '../..'),
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
    });

    let stdoutData = '';
    let stderrData = '';

    pyProcess.stdout.on('data', (chunk) => {
      stdoutData += chunk.toString('utf-8');
    });

    pyProcess.stderr.on('data', (chunk) => {
      stderrData += chunk.toString('utf-8');
    });

    pyProcess.on('close', (code) => {
      if (code !== 0 && !stdoutData.trim()) {
        return reject(new Error(`Python process exited with code ${code}. Error: ${stderrData}`));
      }

      try {
        const firstBrace = stdoutData.indexOf('{');
        const lastBrace = stdoutData.lastIndexOf('}');

        if (firstBrace === -1 || lastBrace === -1) {
          throw new Error(`Invalid JSON output from Python script. Raw output: ${stdoutData}`);
        }

        const jsonStr = stdoutData.slice(firstBrace, lastBrace + 1);
        const parsed = JSON.parse(jsonStr);
        resolve(parsed);
      } catch (err) {
        console.error('Failed to parse Python ML output:', err.message);
        console.error('Raw stdout:', stdoutData);
        console.error('Raw stderr:', stderrData);
        reject(err);
      }
    });

    pyProcess.on('error', (err) => {
      reject(err);
    });
  });
}
