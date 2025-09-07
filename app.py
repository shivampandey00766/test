"""
Web API interface for the 2D to 3D Architectural Conversion AI Agent.
Provides RESTful endpoints for uploading floor plans and downloading 3D models.
"""

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import os
import uuid
import logging
from pathlib import Path
from werkzeug.utils import secure_filename
import json
from typing import Dict, Any
import time

# Import the AI Agent
from src.agent import AIAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize AI Agent
ai_agent = AIAgent()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2D to 3D Architectural Conversion</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.8;
            font-size: 1.1em;
        }
        .content {
            padding: 40px;
        }
        .upload-section {
            border: 3px dashed #3498db;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        .upload-section:hover {
            border-color: #2980b9;
            background-color: #f8f9fa;
        }
        .upload-section.dragover {
            border-color: #27ae60;
            background-color: #e8f5e8;
        }
        .file-input {
            display: none;
        }
        .upload-btn {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px;
        }
        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        .convert-btn {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 25px;
            font-size: 1.2em;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 20px 0;
        }
        .convert-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(39, 174, 96, 0.4);
        }
        .convert-btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
            display: none;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        .status {
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            display: none;
        }
        .status.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .status.info {
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        .download-btn {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 20px;
            text-decoration: none;
            display: inline-block;
            margin: 5px;
            transition: all 0.3s ease;
        }
        .download-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(231, 76, 60, 0.4);
        }
        .preview {
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }
        .feature {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .feature-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏗️ AI Architectural Converter</h1>
            <p>Transform your 2D floor plans into stunning 3D models with AI</p>
        </div>
        
        <div class="content">
            <div class="features">
                <div class="feature">
                    <div class="feature-icon">🎯</div>
                    <h3>Smart Detection</h3>
                    <p>AI-powered recognition of walls, doors, windows, and rooms</p>
                </div>
                <div class="feature">
                    <div class="feature-icon">🧠</div>
                    <h3>Deep Learning</h3>
                    <p>Advanced neural networks for accurate 3D reconstruction</p>
                </div>
                <div class="feature">
                    <div class="feature-icon">📐</div>
                    <h3>Precise Geometry</h3>
                    <p>Vector-based processing for accurate measurements</p>
                </div>
                <div class="feature">
                    <div class="feature-icon">🎨</div>
                    <h3>Realistic Rendering</h3>
                    <p>High-quality 3D models with materials and lighting</p>
                </div>
            </div>
            
            <div class="upload-section" id="uploadSection">
                <h2>📁 Upload Your Floor Plan</h2>
                <p>Drag and drop your floor plan image here, or click to browse</p>
                <input type="file" id="fileInput" class="file-input" accept="image/*">
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    Choose File
                </button>
                <p style="margin-top: 15px; color: #7f8c8d;">
                    Supported formats: PNG, JPG, JPEG, BMP, TIFF (Max 16MB)
                </p>
            </div>
            
            <div id="preview" style="display: none;">
                <h3>📷 Preview</h3>
                <img id="previewImg" class="preview" alt="Floor plan preview">
            </div>
            
            <div class="progress-bar" id="progressBar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            
            <div class="status" id="status"></div>
            
            <div style="text-align: center;">
                <button class="convert-btn" id="convertBtn" onclick="convertTo3D()" disabled>
                    🚀 Convert to 3D
                </button>
            </div>
            
            <div class="results" id="results">
                <h3>🎉 Conversion Complete!</h3>
                <p>Your 3D model has been generated successfully. Download it in your preferred format:</p>
                <div id="downloadLinks"></div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;
        
        // File input handling
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                handleFileSelect(file);
            }
        });
        
        // Drag and drop handling
        const uploadSection = document.getElementById('uploadSection');
        
        uploadSection.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) {
                handleFileSelect(file);
            }
        });
        
        function handleFileSelect(file) {
            // Validate file type
            const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
            if (!allowedTypes.includes(file.type)) {
                showStatus('Please select a valid image file (PNG, JPG, JPEG, BMP, TIFF)', 'error');
                return;
            }
            
            // Validate file size (16MB)
            if (file.size > 16 * 1024 * 1024) {
                showStatus('File size must be less than 16MB', 'error');
                return;
            }
            
            selectedFile = file;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('previewImg').src = e.target.result;
                document.getElementById('preview').style.display = 'block';
            };
            reader.readAsDataURL(file);
            
            // Enable convert button
            document.getElementById('convertBtn').disabled = false;
            
            showStatus('File selected successfully! Ready to convert.', 'success');
        }
        
        async function convertTo3D() {
            if (!selectedFile) {
                showStatus('Please select a file first', 'error');
                return;
            }
            
            const convertBtn = document.getElementById('convertBtn');
            const progressBar = document.getElementById('progressBar');
            const progressFill = document.getElementById('progressFill');
            const results = document.getElementById('results');
            
            // Disable button and show progress
            convertBtn.disabled = true;
            convertBtn.textContent = '🔄 Converting...';
            progressBar.style.display = 'block';
            results.style.display = 'none';
            
            // Create form data
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            try {
                // Simulate progress
                let progress = 0;
                const progressInterval = setInterval(() => {
                    progress += Math.random() * 10;
                    if (progress > 90) progress = 90;
                    progressFill.style.width = progress + '%';
                }, 500);
                
                // Send conversion request
                const response = await fetch('/api/convert', {
                    method: 'POST',
                    body: formData
                });
                
                clearInterval(progressInterval);
                progressFill.style.width = '100%';
                
                if (response.ok) {
                    const result = await response.json();
                    showStatus('Conversion completed successfully!', 'success');
                    showResults(result);
                } else {
                    const error = await response.json();
                    showStatus('Conversion failed: ' + error.error, 'error');
                }
            } catch (error) {
                showStatus('Conversion failed: ' + error.message, 'error');
            } finally {
                convertBtn.disabled = false;
                convertBtn.textContent = '🚀 Convert to 3D';
                setTimeout(() => {
                    progressBar.style.display = 'none';
                    progressFill.style.width = '0%';
                }, 2000);
            }
        }
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
            status.style.display = 'block';
            
            if (type === 'success') {
                setTimeout(() => {
                    status.style.display = 'none';
                }, 5000);
            }
        }
        
        function showResults(result) {
            const results = document.getElementById('results');
            const downloadLinks = document.getElementById('downloadLinks');
            
            let linksHtml = '';
            if (result.exported_models) {
                for (const [format, path] of Object.entries(result.exported_models)) {
                    linksHtml += `<a href="/api/download/${path.split('/').pop()}" class="download-btn">Download ${format.toUpperCase()}</a>`;
                }
            }
            if (result.visualization) {
                linksHtml += `<a href="/api/download/${result.visualization.split('/').pop()}" class="download-btn">Download Visualization</a>`;
            }
            
            downloadLinks.innerHTML = linksHtml;
            results.style.display = 'block';
        }
    </script>
</body>
</html>
"""


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status and AI Agent status."""
    try:
        agent_status = ai_agent.get_status()
        return jsonify({
            'status': 'online',
            'ai_agent': agent_status,
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/convert', methods=['POST'])
def convert_floor_plan():
    """Convert uploaded floor plan to 3D model."""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Create session directories
        session_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        session_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        os.makedirs(session_upload_dir, exist_ok=True)
        os.makedirs(session_output_dir, exist_ok=True)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(session_upload_dir, filename)
        file.save(file_path)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Convert to 3D
        result = ai_agent.convert_2d_to_3d(
            image_path=file_path,
            output_dir=session_output_dir,
            output_name=session_id
        )
        
        if result['success']:
            # Update file paths to be relative to the output directory
            for format_name, path in result['exported_models'].items():
                result['exported_models'][format_name] = os.path.relpath(path, app.config['OUTPUT_FOLDER'])
            
            if result['visualization']:
                result['visualization'] = os.path.relpath(result['visualization'], app.config['OUTPUT_FOLDER'])
            
            return jsonify(result)
        else:
            return jsonify({'error': result.get('error', 'Conversion failed')}), 500
    
    except Exception as e:
        logger.error(f"Error in conversion: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download generated files."""
    try:
        # Find the file in the outputs directory
        for root, dirs, files in os.walk(app.config['OUTPUT_FOLDER']):
            if filename in files:
                file_path = os.path.join(root, filename)
                return send_file(file_path, as_attachment=True)
        
        return jsonify({'error': 'File not found'}), 404
    
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting AI Architectural Converter API")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Output folder: {app.config['OUTPUT_FOLDER']}")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)