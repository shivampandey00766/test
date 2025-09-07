"""
Main AI Agent for 2D to 3D architectural conversion.

This module provides the core AI agent that orchestrates the entire
pipeline from 2D floor plan analysis to 3D model generation.
"""

import os
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from datetime import datetime

# Import all pipeline modules
from ..preprocessing import ImageCleaner, FeatureDetector, PerspectiveCorrector
from ..segmentation import FloorPlanSegmenter, RoomClassifier, DepthEstimator
from ..vectorization import RasterToVectorConverter, GeometryOptimizer, SVGExporter
from ..reconstruction import BlenderReconstructor, Open3DReconstructor, MeshGenerator


class ArchitecturalAgent:
    """
    Main AI Agent for 2D to 3D architectural conversion.
    
    This agent orchestrates the entire pipeline from 2D floor plan
    analysis to 3D model generation, providing a unified interface
    for architectural conversion tasks.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 output_dir: str = "output",
                 log_level: str = "INFO"):
        """
        Initialize the ArchitecturalAgent.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for generated files
            log_level: Logging level
        """
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Configuration
        self.config = config or self._get_default_config()
        
        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize pipeline components
        self._initialize_components()
        
        # Pipeline state
        self.current_session = None
        self.processing_steps = []
        
        self.logger.info("ArchitecturalAgent initialized successfully")
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("ArchitecturalAgent")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'preprocessing': {
                'denoise_method': 'bilateral',
                'contrast_method': 'clahe',
                'binarize_method': 'adaptive',
                'perspective_correction': True
            },
            'segmentation': {
                'use_cnn': True,
                'use_heuristic': True,
                'confidence_threshold': 0.5
            },
            'vectorization': {
                'min_contour_area': 100,
                'approximation_epsilon': 0.02,
                'min_line_length': 10.0,
                'optimize_geometry': True
            },
            'reconstruction': {
                'wall_height': 2.4,
                'floor_thickness': 0.2,
                'ceiling_height': 2.7,
                'reconstruction_method': 'blender'  # 'blender', 'open3d', 'mesh'
            },
            'export': {
                'formats': ['obj', 'gltf', 'svg'],
                'include_metadata': True,
                'generate_visualizations': True
            }
        }
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Preprocessing components
            self.image_cleaner = ImageCleaner()
            self.feature_detector = FeatureDetector()
            self.perspective_corrector = PerspectiveCorrector()
            
            # Segmentation components
            self.segmenter = FloorPlanSegmenter()
            self.room_classifier = RoomClassifier()
            self.depth_estimator = DepthEstimator()
            
            # Vectorization components
            self.vectorizer = RasterToVectorConverter()
            self.geometry_optimizer = GeometryOptimizer()
            self.svg_exporter = SVGExporter()
            
            # Reconstruction components
            reconstruction_config = self.config['reconstruction']
            self.blender_reconstructor = BlenderReconstructor(
                wall_height=reconstruction_config['wall_height'],
                floor_thickness=reconstruction_config['floor_thickness'],
                ceiling_height=reconstruction_config['ceiling_height']
            )
            self.open3d_reconstructor = Open3DReconstructor(
                wall_height=reconstruction_config['wall_height'],
                floor_thickness=reconstruction_config['floor_thickness'],
                ceiling_height=reconstruction_config['ceiling_height']
            )
            self.mesh_generator = MeshGenerator(
                wall_height=reconstruction_config['wall_height'],
                floor_thickness=reconstruction_config['floor_thickness'],
                ceiling_height=reconstruction_config['ceiling_height']
            )
            
            self.logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new processing session.
        
        Args:
            session_id: Optional session ID (generated if None)
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = {
            'id': session_id,
            'start_time': datetime.now(),
            'steps': [],
            'input_files': [],
            'output_files': [],
            'metadata': {}
        }
        
        # Create session directory
        session_dir = self.output_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Started new session: {session_id}")
        return session_id
    
    def process_floor_plan(self, 
                          image_path: str,
                          output_formats: List[str] = None,
                          room_types: List[str] = None,
                          custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a 2D floor plan and generate 3D model.
        
        Args:
            image_path: Path to input floor plan image
            output_formats: List of output formats ['obj', 'gltf', 'svg', 'blend']
            room_types: List of room types for classification
            custom_config: Custom configuration overrides
            
        Returns:
            Dictionary with processing results and file paths
        """
        if self.current_session is None:
            self.start_session()
        
        # Merge custom config
        config = self.config.copy()
        if custom_config:
            config.update(custom_config)
        
        # Set default output formats
        if output_formats is None:
            output_formats = config['export']['formats']
        
        self.logger.info(f"Starting processing of: {image_path}")
        
        try:
            # Step 1: Image Preprocessing
            self.logger.info("Step 1: Image Preprocessing")
            preprocessed_image = self._preprocess_image(image_path, config)
            self._add_step("preprocessing", "completed", {"input": image_path})
            
            # Step 2: Feature Detection
            self.logger.info("Step 2: Feature Detection")
            features = self._detect_features(preprocessed_image, config)
            self._add_step("feature_detection", "completed", {"features_count": len(features)})
            
            # Step 3: Semantic Segmentation
            self.logger.info("Step 3: Semantic Segmentation")
            segmentation_mask = self._segment_image(preprocessed_image, config)
            self._add_step("segmentation", "completed", {"mask_shape": segmentation_mask.shape})
            
            # Step 4: Room Classification
            self.logger.info("Step 4: Room Classification")
            classified_rooms = self._classify_rooms(segmentation_mask, room_types, config)
            self._add_step("room_classification", "completed", {"rooms_count": len(classified_rooms)})
            
            # Step 5: Depth Estimation
            self.logger.info("Step 5: Depth Estimation")
            depth_data = self._estimate_depth(preprocessed_image, segmentation_mask, config)
            self._add_step("depth_estimation", "completed", {"depth_maps": list(depth_data.keys())})
            
            # Step 6: Vectorization
            self.logger.info("Step 6: Vectorization")
            vector_data = self._vectorize_data(segmentation_mask, config)
            self._add_step("vectorization", "completed", {
                "lines": len(vector_data.lines),
                "polygons": len(vector_data.polygons)
            })
            
            # Step 7: 3D Reconstruction
            self.logger.info("Step 7: 3D Reconstruction")
            reconstruction_results = self._reconstruct_3d(
                vector_data, classified_rooms, depth_data, config
            )
            self._add_step("3d_reconstruction", "completed", {
                "method": config['reconstruction']['reconstruction_method']
            })
            
            # Step 8: Export Results
            self.logger.info("Step 8: Export Results")
            output_files = self._export_results(
                vector_data, reconstruction_results, output_formats, config
            )
            self._add_step("export", "completed", {"output_files": output_files})
            
            # Generate visualizations
            if config['export']['generate_visualizations']:
                self.logger.info("Generating Visualizations")
                viz_files = self._generate_visualizations(
                    preprocessed_image, segmentation_mask, vector_data, reconstruction_results
                )
                output_files.update(viz_files)
            
            # Update session
            self.current_session['output_files'].extend(list(output_files.values()))
            self.current_session['end_time'] = datetime.now()
            
            self.logger.info(f"Processing completed successfully. Session: {self.current_session['id']}")
            
            return {
                'session_id': self.current_session['id'],
                'status': 'success',
                'output_files': output_files,
                'processing_time': str(self.current_session['end_time'] - self.current_session['start_time']),
                'steps': self.current_session['steps']
            }
            
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            self._add_step("error", "failed", {"error": str(e)})
            return {
                'session_id': self.current_session['id'],
                'status': 'error',
                'error': str(e),
                'steps': self.current_session['steps']
            }
    
    def _preprocess_image(self, image_path: str, config: Dict[str, Any]) -> np.ndarray:
        """Preprocess the input image."""
        # Load and preprocess image
        preprocessed = self.image_cleaner.preprocess_pipeline(
            image_path,
            denoise_method=config['preprocessing']['denoise_method'],
            contrast_method=config['preprocessing']['contrast_method'],
            binarize_method=config['preprocessing']['binarize_method']
        )
        
        # Perspective correction if enabled
        if config['preprocessing']['perspective_correction']:
            corrected, success = self.perspective_corrector.auto_correct_perspective(preprocessed)
            if success:
                preprocessed = corrected
                self.logger.info("Perspective correction applied")
        
        return preprocessed
    
    def _detect_features(self, image: np.ndarray, config: Dict[str, Any]) -> Dict[str, List]:
        """Detect architectural features in the image."""
        walls = self.feature_detector.detect_walls(image)
        doors = self.feature_detector.detect_doors(image, walls)
        windows = self.feature_detector.detect_windows(image, walls)
        rooms = self.feature_detector.detect_rooms(image)
        
        return {
            'walls': walls,
            'doors': doors,
            'windows': windows,
            'rooms': rooms
        }
    
    def _segment_image(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Perform semantic segmentation on the image."""
        segmentation_mask = self.segmenter.predict(image)
        return segmentation_mask
    
    def _classify_rooms(self, segmentation_mask: np.ndarray, 
                       room_types: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Classify rooms in the segmentation mask."""
        rooms = self.room_classifier.classify_rooms_in_plan(segmentation_mask)
        
        # Convert to dictionary format
        classified_rooms = []
        for i, room in enumerate(rooms):
            room_data = {
                'id': i,
                'type': room.room_type,
                'confidence': room.confidence,
                'area': room.area,
                'center': room.center,
                'features': room.features
            }
            classified_rooms.append(room_data)
        
        return classified_rooms
    
    def _estimate_depth(self, image: np.ndarray, segmentation_mask: np.ndarray, 
                       config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Estimate depth information for 3D reconstruction."""
        room_types = [room['type'] for room in self.current_session.get('classified_rooms', [])]
        
        depth_data = self.depth_estimator.estimate_depth(
            image, segmentation_mask, room_types, 
            use_cnn=config['segmentation']['use_cnn']
        )
        
        return depth_data
    
    def _vectorize_data(self, segmentation_mask: np.ndarray, config: Dict[str, Any]) -> Any:
        """Convert segmentation mask to vector data."""
        vector_data = self.vectorizer.convert_to_vector(segmentation_mask)
        
        # Optimize geometry if enabled
        if config['vectorization']['optimize_geometry']:
            optimization_result = self.geometry_optimizer.optimize_geometry(
                vector_data.lines, vector_data.polygons
            )
            vector_data.lines = optimization_result.optimized_lines
            vector_data.polygons = optimization_result.optimized_polygons
        
        return vector_data
    
    def _reconstruct_3d(self, vector_data: Any, classified_rooms: List[Dict[str, Any]], 
                       depth_data: Dict[str, np.ndarray], config: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct 3D model from vector data."""
        method = config['reconstruction']['reconstruction_method']
        room_types = [room['type'] for room in classified_rooms]
        
        if method == 'blender':
            # Use Blender for reconstruction
            objects = self.blender_reconstructor.reconstruct_from_vector_data(
                vector_data, room_types
            )
            return {'method': 'blender', 'objects': objects}
        
        elif method == 'open3d':
            # Use Open3D for reconstruction
            mesh = self.open3d_reconstructor.reconstruct_from_vector_data(
                vector_data, room_types
            )
            return {'method': 'open3d', 'mesh': mesh}
        
        elif method == 'mesh':
            # Use MeshGenerator for reconstruction
            mesh = self.mesh_generator.combine_meshes([
                self.mesh_generator.create_floor_mesh([p for p in vector_data.polygons if p.polygon_type == 'room']),
                self.mesh_generator.create_ceiling_mesh([p for p in vector_data.polygons if p.polygon_type == 'room'])
            ])
            return {'method': 'mesh', 'mesh': mesh}
        
        else:
            raise ValueError(f"Unsupported reconstruction method: {method}")
    
    def _export_results(self, vector_data: Any, reconstruction_results: Dict[str, Any], 
                       output_formats: List[str], config: Dict[str, Any]) -> Dict[str, str]:
        """Export results in specified formats."""
        output_files = {}
        session_dir = self.output_dir / self.current_session['id']
        
        # Export vector data
        if 'svg' in output_formats:
            svg_path = session_dir / "floor_plan.svg"
            self.svg_exporter.export_to_svg(vector_data, str(svg_path))
            output_files['svg'] = str(svg_path)
        
        # Export 3D model
        method = reconstruction_results['method']
        
        if method == 'blender':
            # Export using Blender
            for format in output_formats:
                if format in ['obj', 'fbx', 'gltf', 'blend']:
                    output_path = session_dir / f"model.{format}"
                    self.blender_reconstructor.export_model(str(output_path), format)
                    output_files[f'model_{format}'] = str(output_path)
        
        elif method in ['open3d', 'mesh']:
            # Export using Open3D or MeshGenerator
            mesh = reconstruction_results['mesh']
            
            for format in output_formats:
                if format in ['obj', 'ply', 'stl', 'gltf']:
                    output_path = session_dir / f"model.{format}"
                    if method == 'open3d':
                        self.open3d_reconstructor.save_model(mesh, str(output_path), format)
                    else:
                        self.mesh_generator.export_mesh(mesh, str(output_path), format)
                    output_files[f'model_{format}'] = str(output_path)
        
        # Export metadata
        if config['export']['include_metadata']:
            metadata_path = session_dir / "metadata.json"
            metadata = {
                'session_id': self.current_session['id'],
                'processing_steps': self.current_session['steps'],
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            output_files['metadata'] = str(metadata_path)
        
        return output_files
    
    def _generate_visualizations(self, preprocessed_image: np.ndarray, 
                                segmentation_mask: np.ndarray, vector_data: Any, 
                                reconstruction_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualization images."""
        session_dir = self.output_dir / self.current_session['id']
        viz_files = {}
        
        # Preprocessing visualization
        viz_path = session_dir / "preprocessing_steps.png"
        self.image_cleaner.visualize_preprocessing_steps(
            str(session_dir / "input_image.jpg"), str(viz_path)
        )
        viz_files['preprocessing'] = str(viz_path)
        
        # Segmentation visualization
        viz_path = session_dir / "segmentation_result.png"
        self.segmenter.visualize_segmentation(
            preprocessed_image, segmentation_mask, str(viz_path)
        )
        viz_files['segmentation'] = str(viz_path)
        
        # Vector data visualization
        viz_path = session_dir / "vector_data.png"
        self.vectorizer.visualize_vector_data(
            vector_data, save_path=str(viz_path)
        )
        viz_files['vector_data'] = str(viz_path)
        
        return viz_files
    
    def _add_step(self, step_name: str, status: str, data: Dict[str, Any]):
        """Add a processing step to the current session."""
        step = {
            'name': step_name,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        self.current_session['steps'].append(step)
        self.processing_steps.append(step)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a processing session."""
        # This would typically load from a database or file
        # For now, return current session if it matches
        if self.current_session and self.current_session['id'] == session_id:
            return self.current_session
        return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all processing sessions."""
        # This would typically load from a database
        # For now, return current session if available
        if self.current_session:
            return [self.current_session]
        return []
    
    def cleanup_session(self, session_id: str) -> bool:
        """Clean up a processing session."""
        try:
            session_dir = self.output_dir / session_id
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)
                self.logger.info(f"Cleaned up session: {session_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error cleaning up session {session_id}: {e}")
            return False