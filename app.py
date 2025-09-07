"""
Flask web application for the AI Agent for 2D to 3D Architectural Conversion.

This module provides a REST API and web interface for uploading floor plans
and generating 3D architectural models.
"""

from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from flask_cors import CORS
import os
import uuid
import json
from pathlib import Path
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import threading
import time

# Import our modules
from src.agent import ArchitecturalAgent, PipelineManager
from src.agent.pipeline_manager import PipelineStatus


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create necessary directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

# Initialize components
agent = ArchitecturalAgent(output_dir=app.config['OUTPUT_FOLDER'])
pipeline_manager = PipelineManager(jobs_dir='jobs')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload a floor plan image file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save file
            file.save(file_path)
            
            logger.info(f"File uploaded: {unique_filename}")
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': unique_filename,
                'file_path': file_path
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process', methods=['POST'])
def process_floor_plan():
    """Process a floor plan and generate 3D model."""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'filename' not in data:
            return jsonify({'error': 'Filename required'}), 400
        
        filename = data['filename']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Get optional parameters
        output_formats = data.get('output_formats', ['obj', 'gltf', 'svg'])
        room_types = data.get('room_types', None)
        custom_config = data.get('config', None)
        
        # Create job
        job_id = pipeline_manager.create_job(
            input_file=file_path,
            output_dir=app.config['OUTPUT_FOLDER'],
            config=custom_config or {}
        )
        
        # Start processing in background
        success = pipeline_manager.start_job(job_id, agent)
        
        if not success:
            return jsonify({'error': 'Failed to start processing'}), 500
        
        return jsonify({
            'message': 'Processing started',
            'job_id': job_id,
            'status': 'running'
        })
    
    except Exception as e:
        logger.error(f"Error processing floor plan: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a processing job."""
    try:
        status = pipeline_manager.get_job_status(job_id)
        
        if not status:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all processing jobs."""
    try:
        status_filter = request.args.get('status')
        if status_filter:
            status_filter = PipelineStatus(status_filter)
        
        jobs = pipeline_manager.list_jobs(status_filter)
        return jsonify({'jobs': jobs})
    
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/job/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Cancel a processing job."""
    try:
        success = pipeline_manager.cancel_job(job_id)
        
        if not success:
            return jsonify({'error': 'Job not found or cannot be cancelled'}), 404
        
        return jsonify({'message': 'Job cancelled successfully'})
    
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/job/<job_id>/download/<file_type>', methods=['GET'])
def download_file(job_id, file_type):
    """Download a generated file."""
    try:
        # Find the file in the job output directory
        job_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
        
        if not job_dir.exists():
            return jsonify({'error': 'Job output not found'}), 404
        
        # Look for files with the requested type
        files = list(job_dir.glob(f"*.{file_type}"))
        
        if not files:
            return jsonify({'error': f'No {file_type} file found for this job'}), 404
        
        # Return the first matching file
        file_path = files[0]
        
        return send_file(
            str(file_path),
            as_attachment=True,
            download_name=f"{job_id}.{file_type}"
        )
    
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/job/<job_id>/files', methods=['GET'])
def list_job_files(job_id):
    """List all files generated for a job."""
    try:
        job_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
        
        if not job_dir.exists():
            return jsonify({'error': 'Job output not found'}), 404
        
        files = []
        for file_path in job_dir.iterdir():
            if file_path.is_file():
                files.append({
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'type': file_path.suffix[1:],  # Remove the dot
                    'created': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
                })
        
        return jsonify({'files': files})
    
    except Exception as e:
        logger.error(f"Error listing job files: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_jobs': len(pipeline_manager.active_jobs)
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    return jsonify({
        'max_file_size': app.config['MAX_CONTENT_LENGTH'],
        'allowed_extensions': list(app.config['ALLOWED_EXTENSIONS']),
        'max_concurrent_jobs': pipeline_manager.max_concurrent_jobs
    })


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration."""
    try:
        data = request.get_json()
        
        # Update allowed file extensions
        if 'allowed_extensions' in data:
            app.config['ALLOWED_EXTENSIONS'] = set(data['allowed_extensions'])
        
        # Update max concurrent jobs
        if 'max_concurrent_jobs' in data:
            pipeline_manager.max_concurrent_jobs = data['max_concurrent_jobs']
        
        return jsonify({'message': 'Configuration updated successfully'})
    
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cleanup/<job_id>', methods=['DELETE'])
def cleanup_job(job_id):
    """Clean up job data and files."""
    try:
        success = pipeline_manager.cleanup_job(job_id)
        
        if not success:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify({'message': 'Job cleaned up successfully'})
    
    except Exception as e:
        logger.error(f"Error cleaning up job: {e}")
        return jsonify({'error': str(e)}), 500


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
    # Create templates directory if it doesn't exist
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Create basic HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Agent for 2D to 3D Architectural Conversion</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: border-color 0.3s;
        }
        .upload-area:hover {
            border-color: #007bff;
        }
        .upload-area.dragover {
            border-color: #007bff;
            background-color: #f8f9fa;
        }
        input[type="file"] {
            margin: 20px 0;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
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
        .progress {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-bar {
            height: 100%;
            background-color: #007bff;
            width: 0%;
            transition: width 0.3s;
        }
        .files {
            margin-top: 20px;
        }
        .file-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 5px 0;
        }
        .file-item a {
            color: #007bff;
            text-decoration: none;
        }
        .file-item a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Agent for 2D to 3D Architectural Conversion</h1>
        
        <div class="upload-area" id="uploadArea">
            <h3>Upload Floor Plan</h3>
            <p>Drag and drop your floor plan image here, or click to select a file</p>
            <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.bmp,.tiff" style="display: none;">
            <button onclick="document.getElementById('fileInput').click()">Choose File</button>
        </div>
        
        <div style="text-align: center;">
            <button id="processBtn" onclick="processFile()" disabled>Process Floor Plan</button>
        </div>
        
        <div class="status" id="status"></div>
        <div class="progress" id="progress" style="display: none;">
            <div class="progress-bar" id="progressBar"></div>
        </div>
        
        <div class="files" id="files" style="display: none;">
            <h3>Generated Files</h3>
            <div id="fileList"></div>
        </div>
    </div>

    <script>
        let currentJobId = null;
        let pollInterval = null;

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const processBtn = document.getElementById('processBtn');
        const status = document.getElementById('status');
        const progress = document.getElementById('progress');
        const progressBar = document.getElementById('progressBar');
        const files = document.getElementById('files');
        const fileList = document.getElementById('fileList');

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        function handleFileSelect(file) {
            fileInput.files = e.target.files;
            processBtn.disabled = false;
            showStatus('File selected: ' + file.name, 'info');
        }

        function showStatus(message, type) {
            status.textContent = message;
            status.className = 'status ' + type;
            status.style.display = 'block';
        }

        function hideStatus() {
            status.style.display = 'none';
        }

        async function processFile() {
            const file = fileInput.files[0];
            if (!file) {
                showStatus('Please select a file first', 'error');
                return;
            }

            try {
                // Upload file
                const formData = new FormData();
                formData.append('file', file);

                const uploadResponse = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!uploadResponse.ok) {
                    const error = await uploadResponse.json();
                    throw new Error(error.error || 'Upload failed');
                }

                const uploadData = await uploadResponse.json();
                showStatus('File uploaded successfully. Starting processing...', 'info');

                // Start processing
                const processResponse = await fetch('/api/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        filename: uploadData.filename,
                        output_formats: ['obj', 'gltf', 'svg']
                    })
                });

                if (!processResponse.ok) {
                    const error = await processResponse.json();
                    throw new Error(error.error || 'Processing failed');
                }

                const processData = await processResponse.json();
                currentJobId = processData.job_id;
                
                showStatus('Processing started. Please wait...', 'info');
                progress.style.display = 'block';
                processBtn.disabled = true;

                // Start polling for status
                pollInterval = setInterval(pollJobStatus, 2000);

            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
                processBtn.disabled = false;
            }
        }

        async function pollJobStatus() {
            if (!currentJobId) return;

            try {
                const response = await fetch(`/api/job/${currentJobId}`);
                if (!response.ok) {
                    throw new Error('Failed to get job status');
                }

                const jobStatus = await response.json();
                
                // Update progress bar
                progressBar.style.width = jobStatus.progress + '%';
                
                if (jobStatus.status === 'completed') {
                    clearInterval(pollInterval);
                    showStatus('Processing completed successfully!', 'success');
                    progress.style.display = 'none';
                    processBtn.disabled = false;
                    loadJobFiles();
                } else if (jobStatus.status === 'failed') {
                    clearInterval(pollInterval);
                    showStatus('Processing failed. Please try again.', 'error');
                    progress.style.display = 'none';
                    processBtn.disabled = false;
                }

            } catch (error) {
                clearInterval(pollInterval);
                showStatus('Error checking job status: ' + error.message, 'error');
                progress.style.display = 'none';
                processBtn.disabled = false;
            }
        }

        async function loadJobFiles() {
            if (!currentJobId) return;

            try {
                const response = await fetch(`/api/job/${currentJobId}/files`);
                if (!response.ok) {
                    throw new Error('Failed to load job files');
                }

                const data = await response.json();
                
                if (data.files.length > 0) {
                    files.style.display = 'block';
                    fileList.innerHTML = '';
                    
                    data.files.forEach(file => {
                        const fileItem = document.createElement('div');
                        fileItem.className = 'file-item';
                        fileItem.innerHTML = `
                            <span>${file.name} (${formatFileSize(file.size)})</span>
                            <a href="/api/job/${currentJobId}/download/${file.type}" download>Download</a>
                        `;
                        fileList.appendChild(fileItem);
                    });
                }

            } catch (error) {
                console.error('Error loading job files:', error);
            }
        }

        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
    </script>
</body>
</html>'''
    
    # Write the HTML template
    with open('templates/index.html', 'w') as f:
        f.write(html_template)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)