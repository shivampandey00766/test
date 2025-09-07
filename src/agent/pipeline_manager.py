"""
Pipeline management utilities for the architectural conversion agent.

This module provides utilities for managing and monitoring the
architectural conversion pipeline.
"""

import json
import time
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from enum import Enum


class PipelineStatus(Enum):
    """Pipeline status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineStep:
    """Represents a single step in the pipeline."""
    name: str
    status: PipelineStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    data: Dict[str, Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


@dataclass
class PipelineJob:
    """Represents a complete pipeline job."""
    job_id: str
    input_file: str
    output_dir: str
    config: Dict[str, Any]
    status: PipelineStatus
    steps: List[PipelineStep]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration: Optional[float] = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []


class PipelineManager:
    """
    Manages the architectural conversion pipeline.
    
    This class provides functionality for managing pipeline jobs,
    monitoring progress, and handling errors.
    """
    
    def __init__(self, 
                 jobs_dir: str = "jobs",
                 max_concurrent_jobs: int = 3,
                 log_level: str = "INFO"):
        """
        Initialize the PipelineManager.
        
        Args:
            jobs_dir: Directory to store job data
            max_concurrent_jobs: Maximum number of concurrent jobs
            log_level: Logging level
        """
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(exist_ok=True)
        
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs = {}
        self.job_queue = []
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Pipeline step definitions
        self.pipeline_steps = [
            "preprocessing",
            "feature_detection", 
            "segmentation",
            "room_classification",
            "depth_estimation",
            "vectorization",
            "3d_reconstruction",
            "export"
        ]
        
        self.logger.info("PipelineManager initialized")
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("PipelineManager")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def create_job(self, 
                   input_file: str,
                   output_dir: str,
                   config: Dict[str, Any],
                   job_id: Optional[str] = None) -> str:
        """
        Create a new pipeline job.
        
        Args:
            input_file: Path to input file
            output_dir: Output directory
            config: Job configuration
            job_id: Optional job ID (generated if None)
            
        Returns:
            Job ID
        """
        if job_id is None:
            job_id = f"job_{int(time.time())}"
        
        # Create job
        job = PipelineJob(
            job_id=job_id,
            input_file=input_file,
            output_dir=output_dir,
            config=config,
            status=PipelineStatus.PENDING,
            created_at=datetime.now()
        )
        
        # Initialize steps
        for step_name in self.pipeline_steps:
            step = PipelineStep(name=step_name, status=PipelineStatus.PENDING)
            job.steps.append(step)
        
        # Save job
        self._save_job(job)
        
        # Add to queue
        self.job_queue.append(job_id)
        
        self.logger.info(f"Created job: {job_id}")
        return job_id
    
    def start_job(self, job_id: str, agent: Any) -> bool:
        """
        Start a pipeline job.
        
        Args:
            job_id: Job ID to start
            agent: ArchitecturalAgent instance
            
        Returns:
            True if job started successfully
        """
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            self.logger.warning(f"Maximum concurrent jobs reached ({self.max_concurrent_jobs})")
            return False
        
        job = self._load_job(job_id)
        if not job:
            self.logger.error(f"Job not found: {job_id}")
            return False
        
        if job.status != PipelineStatus.PENDING:
            self.logger.error(f"Job {job_id} is not pending (status: {job.status})")
            return False
        
        # Update job status
        job.status = PipelineStatus.RUNNING
        job.started_at = datetime.now()
        self._save_job(job)
        
        # Add to active jobs
        self.active_jobs[job_id] = job
        
        # Start processing in background
        self._process_job_async(job_id, agent)
        
        self.logger.info(f"Started job: {job_id}")
        return True
    
    def _process_job_async(self, job_id: str, agent: Any):
        """Process job asynchronously."""
        try:
            job = self.active_jobs[job_id]
            
            # Process each step
            for step in job.steps:
                if step.status == PipelineStatus.PENDING:
                    self._execute_step(step, job, agent)
                    
                    # Check if job was cancelled
                    if job.status == PipelineStatus.CANCELLED:
                        break
            
            # Mark job as completed
            if job.status == PipelineStatus.RUNNING:
                job.status = PipelineStatus.COMPLETED
                job.completed_at = datetime.now()
                job.total_duration = (job.completed_at - job.started_at).total_seconds()
            
        except Exception as e:
            self.logger.error(f"Error processing job {job_id}: {e}")
            job.status = PipelineStatus.FAILED
            job.completed_at = datetime.now()
            if job.total_duration is None:
                job.total_duration = (job.completed_at - job.started_at).total_seconds()
        
        finally:
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            # Save final job state
            self._save_job(job)
    
    def _execute_step(self, step: PipelineStep, job: PipelineJob, agent: Any):
        """Execute a single pipeline step."""
        step.status = PipelineStatus.RUNNING
        step.start_time = datetime.now()
        
        try:
            # Execute step based on name
            if step.name == "preprocessing":
                result = agent._preprocess_image(job.input_file, job.config)
                step.data['preprocessed_shape'] = result.shape
            
            elif step.name == "feature_detection":
                preprocessed = agent._preprocess_image(job.input_file, job.config)
                result = agent._detect_features(preprocessed, job.config)
                step.data['features_count'] = sum(len(v) for v in result.values())
            
            elif step.name == "segmentation":
                preprocessed = agent._preprocess_image(job.input_file, job.config)
                result = agent._segment_image(preprocessed, job.config)
                step.data['mask_shape'] = result.shape
            
            elif step.name == "room_classification":
                preprocessed = agent._preprocess_image(job.input_file, job.config)
                segmentation_mask = agent._segment_image(preprocessed, job.config)
                result = agent._classify_rooms(segmentation_mask, None, job.config)
                step.data['rooms_count'] = len(result)
            
            elif step.name == "depth_estimation":
                preprocessed = agent._preprocess_image(job.input_file, job.config)
                segmentation_mask = agent._segment_image(preprocessed, job.config)
                result = agent._estimate_depth(preprocessed, segmentation_mask, job.config)
                step.data['depth_maps'] = list(result.keys())
            
            elif step.name == "vectorization":
                preprocessed = agent._preprocess_image(job.input_file, job.config)
                segmentation_mask = agent._segment_image(preprocessed, job.config)
                result = agent._vectorize_data(segmentation_mask, job.config)
                step.data['lines_count'] = len(result.lines)
                step.data['polygons_count'] = len(result.polygons)
            
            elif step.name == "3d_reconstruction":
                preprocessed = agent._preprocess_image(job.input_file, job.config)
                segmentation_mask = agent._segment_image(preprocessed, job.config)
                classified_rooms = agent._classify_rooms(segmentation_mask, None, job.config)
                depth_data = agent._estimate_depth(preprocessed, segmentation_mask, job.config)
                vector_data = agent._vectorize_data(segmentation_mask, job.config)
                result = agent._reconstruct_3d(vector_data, classified_rooms, depth_data, job.config)
                step.data['reconstruction_method'] = result['method']
            
            elif step.name == "export":
                # This would be handled by the main processing function
                step.data['export_formats'] = job.config['export']['formats']
                result = "exported"
            
            else:
                raise ValueError(f"Unknown step: {step.name}")
            
            # Mark step as completed
            step.status = PipelineStatus.COMPLETED
            step.data['result'] = str(result) if not isinstance(result, (dict, list)) else "completed"
            
        except Exception as e:
            step.status = PipelineStatus.FAILED
            step.error = str(e)
            self.logger.error(f"Step {step.name} failed: {e}")
            raise
        
        finally:
            step.end_time = datetime.now()
            if step.start_time:
                step.duration = (step.end_time - step.start_time).total_seconds()
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pipeline job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if job was cancelled
        """
        job = self._load_job(job_id)
        if not job:
            return False
        
        if job.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED, PipelineStatus.CANCELLED]:
            return False
        
        job.status = PipelineStatus.CANCELLED
        job.completed_at = datetime.now()
        
        if job.started_at:
            job.total_duration = (job.completed_at - job.started_at).total_seconds()
        
        self._save_job(job)
        
        # Remove from active jobs if running
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        self.logger.info(f"Cancelled job: {job_id}")
        return True
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status and progress.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dictionary
        """
        job = self._load_job(job_id)
        if not job:
            return None
        
        # Calculate progress
        completed_steps = sum(1 for step in job.steps if step.status == PipelineStatus.COMPLETED)
        total_steps = len(job.steps)
        progress = (completed_steps / total_steps) * 100 if total_steps > 0 else 0
        
        return {
            'job_id': job_id,
            'status': job.status.value,
            'progress': progress,
            'completed_steps': completed_steps,
            'total_steps': total_steps,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'total_duration': job.total_duration,
            'steps': [asdict(step) for step in job.steps]
        }
    
    def list_jobs(self, status_filter: Optional[PipelineStatus] = None) -> List[Dict[str, Any]]:
        """
        List all jobs with optional status filter.
        
        Args:
            status_filter: Optional status filter
            
        Returns:
            List of job information
        """
        jobs = []
        
        # Load all job files
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                job = self._load_job_from_file(job_file)
                if job and (status_filter is None or job.status == status_filter):
                    jobs.append({
                        'job_id': job.job_id,
                        'status': job.status.value,
                        'input_file': job.input_file,
                        'created_at': job.created_at.isoformat(),
                        'started_at': job.started_at.isoformat() if job.started_at else None,
                        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                        'total_duration': job.total_duration
                    })
            except Exception as e:
                self.logger.error(f"Error loading job from {job_file}: {e}")
        
        return sorted(jobs, key=lambda x: x['created_at'], reverse=True)
    
    def get_job_logs(self, job_id: str) -> List[str]:
        """
        Get logs for a specific job.
        
        Args:
            job_id: Job ID
            
        Returns:
            List of log messages
        """
        log_file = self.jobs_dir / f"{job_id}_logs.txt"
        if log_file.exists():
            with open(log_file, 'r') as f:
                return f.readlines()
        return []
    
    def cleanup_job(self, job_id: str) -> bool:
        """
        Clean up job data and files.
        
        Args:
            job_id: Job ID to clean up
            
        Returns:
            True if cleaned up successfully
        """
        try:
            # Remove job file
            job_file = self.jobs_dir / f"{job_id}.json"
            if job_file.exists():
                job_file.unlink()
            
            # Remove log file
            log_file = self.jobs_dir / f"{job_id}_logs.txt"
            if log_file.exists():
                log_file.unlink()
            
            self.logger.info(f"Cleaned up job: {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up job {job_id}: {e}")
            return False
    
    def _save_job(self, job: PipelineJob):
        """Save job to file."""
        job_file = self.jobs_dir / f"{job.job_id}.json"
        
        # Convert to dictionary
        job_dict = asdict(job)
        
        # Convert datetime objects to strings
        for key, value in job_dict.items():
            if isinstance(value, datetime):
                job_dict[key] = value.isoformat()
            elif key == 'steps' and isinstance(value, list):
                for step in value:
                    for step_key, step_value in step.items():
                        if isinstance(step_value, datetime):
                            step[step_key] = step_value.isoformat()
        
        with open(job_file, 'w') as f:
            json.dump(job_dict, f, indent=2)
    
    def _load_job(self, job_id: str) -> Optional[PipelineJob]:
        """Load job from file."""
        job_file = self.jobs_dir / f"{job_id}.json"
        return self._load_job_from_file(job_file)
    
    def _load_job_from_file(self, job_file: Path) -> Optional[PipelineJob]:
        """Load job from specific file."""
        if not job_file.exists():
            return None
        
        try:
            with open(job_file, 'r') as f:
                job_dict = json.load(f)
            
            # Convert datetime strings back to datetime objects
            for key, value in job_dict.items():
                if key in ['created_at', 'started_at', 'completed_at'] and value:
                    job_dict[key] = datetime.fromisoformat(value)
                elif key == 'steps' and isinstance(value, list):
                    for step in value:
                        for step_key, step_value in step.items():
                            if step_key in ['start_time', 'end_time'] and step_value:
                                step[step_key] = datetime.fromisoformat(step_value)
            
            # Convert status strings back to enums
            job_dict['status'] = PipelineStatus(job_dict['status'])
            for step in job_dict['steps']:
                step['status'] = PipelineStatus(step['status'])
            
            return PipelineJob(**job_dict)
            
        except Exception as e:
            self.logger.error(f"Error loading job from {job_file}: {e}")
            return None