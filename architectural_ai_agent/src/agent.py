"""
Main AI Agent for 2D to 3D Architectural Conversion

This is the main orchestrator that coordinates all the components to convert
2D floor plans into 3D architectural models.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
import time
import json

# Import all components
from .preprocessing import ImagePreprocessor, Vectorizer
from .detection import FeatureDetector, ObjectDetector
from .segmentation import SemanticSegmentationModel, DepthEstimator
from .reconstruction import Model3DReconstructor, GeometryBuilder, MeshGenerator
from .utils import Config, setup_logging, Visualizer

logger = logging.getLogger(__name__)


class ArchitecturalAIAgent:
    """
    Main AI agent for 2D to 3D architectural conversion.
    
    This class orchestrates the entire pipeline from image preprocessing
    through 3D model generation.
    """
    
    def __init__(self, config: Optional[Union[str, Dict, Config]] = None):
        """
        Initialize the architectural AI agent.
        
        Args:
            config: Configuration (file path, dict, or Config object)
        """
        # Setup configuration
        if isinstance(config, str):
            self.config = Config(config)
        elif isinstance(config, dict):
            self.config = Config()
            self.config.update(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            self.config = Config()
        
        # Setup logging
        log_level = self.config.get('logging.level', 'INFO')
        log_file = self.config.get('paths.logs_dir', None)
        if log_file:
            log_file = Path(log_file) / 'architectural_ai_agent.log'
        
        self.logger = setup_logging(log_level, str(log_file) if log_file else None)
        
        # Initialize components
        self._initialize_components()
        
        # Processing statistics
        self.processing_stats = {}
        
        self.logger.info("Architectural AI Agent initialized successfully")
    
    def _initialize_components(self):
        """Initialize all AI agent components."""
        try:
            # Preprocessing components
            target_size = tuple(self.config.get('preprocessing.target_size', [1024, 1024]))
            self.image_preprocessor = ImagePreprocessor(target_size=target_size)
            self.vectorizer = Vectorizer()
            
            # Detection components
            self.feature_detector = FeatureDetector()
            
            device = self.config.get('device', 'auto')
            self.object_detector = ObjectDetector(device=device)
            
            # Segmentation components
            seg_config = self.config.get('model.segmentation', {})
            self.segmentation_model = SemanticSegmentationModel(
                model_name=seg_config.get('architecture', 'unet'),
                encoder=seg_config.get('encoder', 'resnet50'),
                num_classes=seg_config.get('num_classes', 15),
                device=device
            )
            
            self.depth_estimator = DepthEstimator(device=device)
            
            # Reconstruction components
            output_format = self.config.get('reconstruction.output_format', 'obj')
            self.model_reconstructor = Model3DReconstructor(output_format=output_format)
            self.geometry_builder = GeometryBuilder()
            self.mesh_generator = MeshGenerator()
            
            # Visualization
            self.visualizer = Visualizer()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_floor_plan(self, image_path: str, output_dir: str = None,
                          visualize: bool = True, save_intermediate: bool = False) -> Dict[str, Any]:
        """
        Process a floor plan image and generate a 3D model.
        
        Args:
            image_path: Path to the floor plan image
            output_dir: Directory to save outputs
            visualize: Whether to generate visualizations
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary containing all processing results
        """
        start_time = time.time()
        self.logger.info(f"Starting floor plan processing: {image_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = self.config.get('paths.output_dir', 'output')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results dictionary
        results = {
            'input_path': image_path,
            'output_dir': str(output_path),
            'processing_stages': {},
            'timing': {}
        }
        
        try:
            # Stage 1: Image Preprocessing
            stage_start = time.time()
            self.logger.info("Stage 1: Image preprocessing")
            
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            processed_image = self.image_preprocessor.preprocess_image(image_path)
            results['processing_stages']['preprocessing'] = {
                'original_shape': original_image.shape,
                'processed_shape': processed_image.shape,
                'scale_factor': self.image_preprocessor.get_scale_factor()
            }
            
            stage_time = time.time() - stage_start
            results['timing']['preprocessing'] = stage_time
            self.logger.info(f"Preprocessing completed in {stage_time:.2f}s")
            
            # Stage 2: Vectorization
            stage_start = time.time()
            self.logger.info("Stage 2: Vectorization")
            
            vector_data = self.vectorizer.vectorize_image(processed_image)
            results['processing_stages']['vectorization'] = {
                'num_walls': len(vector_data['walls']),
                'num_thin_lines': len(vector_data['thin_lines']),
                'num_rectangles': len(vector_data['rectangles']),
                'num_circles': len(vector_data['circles'])
            }
            
            stage_time = time.time() - stage_start
            results['timing']['vectorization'] = stage_time
            self.logger.info(f"Vectorization completed in {stage_time:.2f}s")
            
            # Stage 3: Feature Detection
            stage_start = time.time()
            self.logger.info("Stage 3: Feature detection")
            
            features_data = self.feature_detector.detect_features(processed_image, vector_data)
            results['processing_stages']['feature_detection'] = {
                'num_walls': len(features_data['walls']),
                'num_doors': len(features_data['doors']),
                'num_windows': len(features_data['windows']),
                'num_rooms': len(features_data['rooms']),
                'num_fixtures': len(features_data['fixtures'])
            }
            
            stage_time = time.time() - stage_start
            results['timing']['feature_detection'] = stage_time
            self.logger.info(f"Feature detection completed in {stage_time:.2f}s")
            
            # Stage 4: Object Detection (Deep Learning)
            stage_start = time.time()
            self.logger.info("Stage 4: Object detection")
            
            confidence_threshold = self.config.get('model.object_detection.confidence_threshold', 0.5)
            object_detections = self.object_detector.detect_objects(processed_image, confidence_threshold)
            results['processing_stages']['object_detection'] = {
                'num_detections': object_detections['num_detections']
            }
            
            stage_time = time.time() - stage_start
            results['timing']['object_detection'] = stage_time
            self.logger.info(f"Object detection completed in {stage_time:.2f}s")
            
            # Stage 5: Semantic Segmentation
            stage_start = time.time()
            self.logger.info("Stage 5: Semantic segmentation")
            
            segmentation_results = self.segmentation_model.segment_image(processed_image)
            results['processing_stages']['segmentation'] = {
                'num_classes_detected': len(np.unique(segmentation_results['segmentation_map'])),
                'room_analysis': segmentation_results['room_analysis']
            }
            
            stage_time = time.time() - stage_start
            results['timing']['segmentation'] = stage_time
            self.logger.info(f"Segmentation completed in {stage_time:.2f}s")
            
            # Stage 6: Depth Estimation
            stage_start = time.time()
            self.logger.info("Stage 6: Depth estimation")
            
            depth_data = self.depth_estimator.estimate_depths(
                processed_image, segmentation_results, features_data
            )
            results['processing_stages']['depth_estimation'] = {
                'scale_factor': depth_data['scale_factor'],
                'wall_heights': depth_data['wall_heights'],
                'ceiling_heights': depth_data['ceiling_heights']
            }
            
            stage_time = time.time() - stage_start
            results['timing']['depth_estimation'] = stage_time
            self.logger.info(f"Depth estimation completed in {stage_time:.2f}s")
            
            # Stage 7: 3D Reconstruction
            stage_start = time.time()
            self.logger.info("Stage 7: 3D reconstruction")
            
            model_3d = self.model_reconstructor.reconstruct_3d_model(
                segmentation_results, features_data, depth_data, depth_data['scale_factor']
            )
            results['processing_stages']['reconstruction'] = {
                'num_objects': len(model_3d['scene'].geometry) if model_3d['scene'] else 0,
                'total_vertices': model_3d.get('total_vertices', 0),
                'total_faces': model_3d.get('total_faces', 0)
            }
            
            stage_time = time.time() - stage_start
            results['timing']['reconstruction'] = stage_time
            self.logger.info(f"3D reconstruction completed in {stage_time:.2f}s")
            
            # Combine all results
            results.update({
                'original_image': original_image,
                'processed_image': processed_image,
                'vector_data': vector_data,
                'features_data': features_data,
                'object_detections': object_detections,
                'segmentation_results': segmentation_results,
                'depth_data': depth_data,
                'model_3d': model_3d
            })
            
            # Export 3D model
            model_filename = f"{Path(image_path).stem}_3d_model.{self.model_reconstructor.output_format}"
            model_output_path = output_path / model_filename
            
            export_success = self.model_reconstructor.export_model(str(model_output_path), model_3d)
            results['export_success'] = export_success
            results['model_output_path'] = str(model_output_path) if export_success else None
            
            # Generate visualizations
            if visualize:
                self._generate_visualizations(results, output_path)
            
            # Save intermediate results
            if save_intermediate:
                self._save_intermediate_results(results, output_path)
            
            # Calculate total processing time
            total_time = time.time() - start_time
            results['timing']['total'] = total_time
            
            # Generate summary report
            summary_report = self.visualizer.create_summary_report(results)
            results['summary_report'] = summary_report
            
            # Save summary report
            report_path = output_path / f"{Path(image_path).stem}_report.txt"
            with open(report_path, 'w') as f:
                f.write(summary_report)
            
            self.logger.info(f"Floor plan processing completed successfully in {total_time:.2f}s")
            self.logger.info(f"Results saved to: {output_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during floor plan processing: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results
    
    def _generate_visualizations(self, results: Dict, output_path: Path):
        """Generate and save visualizations."""
        try:
            self.logger.info("Generating visualizations...")
            
            # Preprocessing visualization
            if 'original_image' in results and 'processed_image' in results:
                fig = self.visualizer.visualize_preprocessing_results(
                    results['original_image'], 
                    results['processed_image'],
                    str(output_path / 'preprocessing_results.png')
                )
                plt.close(fig)
            
            # Feature detection visualization
            if 'processed_image' in results and 'features_data' in results:
                fig = self.visualizer.visualize_feature_detection(
                    results['processed_image'],
                    results['features_data'],
                    str(output_path / 'feature_detection.png')
                )
                plt.close(fig)
            
            # Segmentation visualization
            if ('processed_image' in results and 'segmentation_results' in results):
                seg_results = results['segmentation_results']
                fig = self.visualizer.visualize_segmentation(
                    results['processed_image'],
                    seg_results['segmentation_map'],
                    seg_results['class_names'],
                    str(output_path / 'segmentation_results.png')
                )
                plt.close(fig)
            
            # Depth estimation visualization
            if 'processed_image' in results and 'depth_data' in results:
                depth_map = results['depth_data'].get('depth_map')
                if depth_map is not None:
                    fig = self.visualizer.visualize_depth_estimation(
                        results['processed_image'],
                        depth_map,
                        str(output_path / 'depth_estimation.png')
                    )
                    plt.close(fig)
            
            # Interactive 3D visualization
            if 'model_3d' in results:
                fig = self.visualizer.visualize_3d_model_interactive(results['model_3d'])
                self.visualizer.save_interactive_plot(
                    fig, str(output_path / '3d_model_interactive.html')
                )
            
            # Room analysis visualization
            if 'segmentation_results' in results:
                room_analysis = results['segmentation_results'].get('room_analysis', {})
                if room_analysis:
                    fig = self.visualizer.visualize_room_analysis(room_analysis)
                    self.visualizer.save_interactive_plot(
                        fig, str(output_path / 'room_analysis.html')
                    )
            
            self.logger.info("Visualizations generated successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate some visualizations: {e}")
    
    def _save_intermediate_results(self, results: Dict, output_path: Path):
        """Save intermediate processing results."""
        try:
            self.logger.info("Saving intermediate results...")
            
            # Save processed image
            if 'processed_image' in results:
                cv2.imwrite(str(output_path / 'processed_image.png'), results['processed_image'])
            
            # Save segmentation map
            if 'segmentation_results' in results:
                seg_map = results['segmentation_results']['segmentation_map']
                cv2.imwrite(str(output_path / 'segmentation_map.png'), seg_map * 15)  # Scale for visibility
            
            # Save depth map
            if 'depth_data' in results:
                depth_map = results['depth_data'].get('depth_map')
                if depth_map is not None:
                    depth_normalized = (depth_map * 255).astype(np.uint8)
                    cv2.imwrite(str(output_path / 'depth_map.png'), depth_normalized)
            
            # Save processing statistics as JSON
            stats = {
                'processing_stages': results.get('processing_stages', {}),
                'timing': results.get('timing', {}),
                'export_success': results.get('export_success', False)
            }
            
            with open(output_path / 'processing_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info("Intermediate results saved successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to save some intermediate results: {e}")
    
    def batch_process(self, input_directory: str, output_directory: str = None,
                     file_pattern: str = "*.png", **kwargs) -> List[Dict]:
        """
        Process multiple floor plan images in batch.
        
        Args:
            input_directory: Directory containing floor plan images
            output_directory: Directory to save outputs
            file_pattern: Pattern to match image files
            **kwargs: Additional arguments passed to process_floor_plan
            
        Returns:
            List of processing results for each image
        """
        input_path = Path(input_directory)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_directory}")
        
        # Find all matching files
        image_files = list(input_path.glob(file_pattern))
        if not image_files:
            self.logger.warning(f"No files found matching pattern {file_pattern} in {input_directory}")
            return []
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Setup output directory
        if output_directory is None:
            output_directory = self.config.get('paths.output_dir', 'output')
        
        output_base = Path(output_directory)
        output_base.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        batch_results = []
        
        for i, image_file in enumerate(image_files):
            self.logger.info(f"Processing image {i+1}/{len(image_files)}: {image_file.name}")
            
            # Create individual output directory
            image_output_dir = output_base / image_file.stem
            
            try:
                result = self.process_floor_plan(
                    str(image_file), 
                    str(image_output_dir),
                    **kwargs
                )
                result['batch_index'] = i
                result['success'] = True
                
            except Exception as e:
                self.logger.error(f"Failed to process {image_file.name}: {e}")
                result = {
                    'input_path': str(image_file),
                    'batch_index': i,
                    'success': False,
                    'error': str(e)
                }
            
            batch_results.append(result)
        
        # Generate batch summary
        successful_count = sum(1 for r in batch_results if r.get('success', False))
        self.logger.info(f"Batch processing completed: {successful_count}/{len(image_files)} successful")
        
        # Save batch summary
        batch_summary = {
            'total_images': len(image_files),
            'successful_count': successful_count,
            'failed_count': len(image_files) - successful_count,
            'results': batch_results
        }
        
        with open(output_base / 'batch_summary.json', 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        return batch_results
    
    def load_trained_models(self, models_directory: str):
        """
        Load pre-trained models from directory.
        
        Args:
            models_directory: Directory containing trained model files
        """
        models_path = Path(models_directory)
        
        if not models_path.exists():
            self.logger.warning(f"Models directory does not exist: {models_directory}")
            return
        
        try:
            # Load segmentation model
            seg_model_path = models_path / 'segmentation_model.pth'
            if seg_model_path.exists():
                self.segmentation_model.load_model(str(seg_model_path))
                self.logger.info("Loaded segmentation model")
            
            # Load object detection model
            obj_model_path = models_path / 'object_detection_model.pth'
            if obj_model_path.exists():
                self.object_detector.load_model(str(obj_model_path))
                self.logger.info("Loaded object detection model")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
    
    def get_processing_statistics(self) -> Dict:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def update_config(self, updates: Dict):
        """Update configuration."""
        self.config.update(updates)
        self.logger.info("Configuration updated")
    
    def save_config(self, config_path: str):
        """Save current configuration to file."""
        self.config.save(config_path)


# Import matplotlib here to avoid issues with backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt