# API Reference

This document provides detailed information about the AI Agent for 2D to 3D Architectural Conversion API.

## Overview

The API provides REST endpoints for uploading floor plan images, processing them through the AI pipeline, and downloading generated 3D models and other outputs.

## Base URL

```
http://localhost:5000/api
```

## Authentication

Currently, no authentication is required. All endpoints are publicly accessible.

## Endpoints

### File Upload

#### POST /api/upload

Upload a floor plan image file.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `file` field containing the image

**Response:**
```json
{
  "message": "File uploaded successfully",
  "filename": "unique_filename.jpg",
  "file_path": "/path/to/uploaded/file"
}
```

**Error Responses:**
- `400`: No file provided or invalid file type
- `413`: File too large (max 16MB)
- `500`: Server error

### Processing

#### POST /api/process

Start processing a floor plan image.

**Request:**
```json
{
  "filename": "unique_filename.jpg",
  "output_formats": ["obj", "gltf", "svg"],
  "room_types": ["living_room", "bedroom", "kitchen"],
  "config": {
    "reconstruction": {
      "wall_height": 2.4,
      "ceiling_height": 2.7
    }
  }
}
```

**Response:**
```json
{
  "message": "Processing started",
  "job_id": "job_1234567890",
  "status": "running"
}
```

### Job Management

#### GET /api/job/{job_id}

Get the status of a processing job.

**Response:**
```json
{
  "job_id": "job_1234567890",
  "status": "running",
  "progress": 45.5,
  "completed_steps": 4,
  "total_steps": 8,
  "created_at": "2024-01-01T12:00:00",
  "started_at": "2024-01-01T12:00:05",
  "completed_at": null,
  "total_duration": null,
  "steps": [
    {
      "name": "preprocessing",
      "status": "completed",
      "start_time": "2024-01-01T12:00:05",
      "end_time": "2024-01-01T12:00:10",
      "duration": 5.0,
      "data": {
        "preprocessed_shape": [512, 512]
      },
      "error": null
    }
  ]
}
```

#### GET /api/jobs

List all processing jobs.

**Query Parameters:**
- `status` (optional): Filter by job status (`pending`, `running`, `completed`, `failed`, `cancelled`)

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "job_1234567890",
      "status": "completed",
      "input_file": "/path/to/input.jpg",
      "created_at": "2024-01-01T12:00:00",
      "started_at": "2024-01-01T12:00:05",
      "completed_at": "2024-01-01T12:05:30",
      "total_duration": 325.0
    }
  ]
}
```

#### POST /api/job/{job_id}/cancel

Cancel a processing job.

**Response:**
```json
{
  "message": "Job cancelled successfully"
}
```

### File Downloads

#### GET /api/job/{job_id}/download/{file_type}

Download a generated file.

**Parameters:**
- `job_id`: Job identifier
- `file_type`: File type (`obj`, `gltf`, `svg`, `blend`, `ply`, `stl`)

**Response:**
- File download

#### GET /api/job/{job_id}/files

List all files generated for a job.

**Response:**
```json
{
  "files": [
    {
      "name": "model.obj",
      "size": 1024000,
      "type": "obj",
      "created": "2024-01-01T12:05:30"
    },
    {
      "name": "model.gltf",
      "size": 512000,
      "type": "gltf",
      "created": "2024-01-01T12:05:30"
    }
  ]
}
```

### System

#### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "active_jobs": 2
}
```

#### GET /api/config

Get current system configuration.

**Response:**
```json
{
  "max_file_size": 16777216,
  "allowed_extensions": ["png", "jpg", "jpeg", "bmp", "tiff"],
  "max_concurrent_jobs": 3
}
```

#### POST /api/config

Update system configuration.

**Request:**
```json
{
  "allowed_extensions": ["png", "jpg", "jpeg"],
  "max_concurrent_jobs": 5
}
```

**Response:**
```json
{
  "message": "Configuration updated successfully"
}
```

### Cleanup

#### DELETE /api/cleanup/{job_id}

Clean up job data and files.

**Response:**
```json
{
  "message": "Job cleaned up successfully"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes and error messages in JSON format:

```json
{
  "error": "Error message describing what went wrong"
}
```

Common status codes:
- `200`: Success
- `400`: Bad request (invalid input)
- `404`: Not found (job or file not found)
- `413`: Payload too large (file too big)
- `500`: Internal server error

## Rate Limiting

Currently, no rate limiting is implemented. However, the system limits the number of concurrent jobs based on the `max_concurrent_jobs` configuration.

## File Formats

### Supported Input Formats
- PNG
- JPEG/JPG
- BMP
- TIFF

### Supported Output Formats
- **OBJ**: Wavefront OBJ format (3D model)
- **GLTF**: glTF format (3D model)
- **SVG**: Scalable Vector Graphics (2D vector)
- **BLEND**: Blender format (3D model)
- **PLY**: Polygon File Format (3D model)
- **STL**: STereoLithography format (3D model)

## Configuration Options

The system can be configured through the `/api/config` endpoint or by modifying the configuration files. Key options include:

- **Preprocessing**: Image cleaning and enhancement methods
- **Segmentation**: Deep learning model settings
- **Vectorization**: Geometry optimization parameters
- **Reconstruction**: 3D model generation settings
- **Export**: Output format and visualization options

## Examples

### Python Client Example

```python
import requests
import time

# Upload file
with open('floor_plan.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/upload', files=files)
    upload_data = response.json()

# Start processing
process_data = {
    'filename': upload_data['filename'],
    'output_formats': ['obj', 'gltf', 'svg']
}
response = requests.post('http://localhost:5000/api/process', json=process_data)
job_data = response.json()
job_id = job_data['job_id']

# Poll for completion
while True:
    response = requests.get(f'http://localhost:5000/api/job/{job_id}')
    job_status = response.json()
    
    if job_status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(2)

# Download results
if job_status['status'] == 'completed':
    response = requests.get(f'http://localhost:5000/api/job/{job_id}/files')
    files = response.json()['files']
    
    for file_info in files:
        response = requests.get(f'http://localhost:5000/api/job/{job_id}/download/{file_info["type"]}')
        with open(file_info['name'], 'wb') as f:
            f.write(response.content)
```

### cURL Examples

```bash
# Upload file
curl -X POST -F "file=@floor_plan.jpg" http://localhost:5000/api/upload

# Start processing
curl -X POST -H "Content-Type: application/json" \
  -d '{"filename":"unique_filename.jpg","output_formats":["obj","gltf"]}' \
  http://localhost:5000/api/process

# Check job status
curl http://localhost:5000/api/job/job_1234567890

# Download file
curl -O http://localhost:5000/api/job/job_1234567890/download/obj
```