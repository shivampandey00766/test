"""
Main AI Agent for 2D to 3D architectural conversion.
Orchestrates the entire pipeline from image input to 3D model output.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
from pathlib import Path

# Import all pipeline components
from ..preprocessing import ImageProcessor, FeatureDetector, NoiseReducer
from ..models import FloorPlanSegmentationModel, DepthEstimationModel, RoomClassifier
from ..vectorization import Vectorizer, GeometryProcessor
from ..reconstruction import ModelBuilder

logger = logging.getLogger(__name__)


class AIAgent:
    """
    Main AI Agent that orchestrates the 2D to 3D conversion pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AI Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipeline components
        self._initialize_components()
        
        # Load models if available
        self._load_models()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the agent."""
        return {
            'image_processing': {
                'target_size': (1024, 1024),
                'noise_reduction_method': 'adaptive',
                'contrast_enhancement_method': 'clahe'
            },
            'models': {
                'segmentation_model_path': 'models/segmentation_model.pth',
                'depth_model_path': 'models/depth_model.pth',
                'room_classifier_path': 'models/room_classifier.pth',
                'device': 'cpu'
            },
            'vectorization': {
                'scale_factor': 1.0,
                'tolerance': 1.0
            },
            '3d_reconstruction': {
                'scale_factor': 0.01,
                'default_wall_height': 2.5,
                'default_door_height': 2.0,
                'default_window_height': 1.5
            },
            'output': {
                'formats': ['obj', 'gltf'],
                'render_resolution': (1920, 1080),
                'export_materials': True,
                'export_lighting': True
            }
        }
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Image processing components
            self.image_processor = ImageProcessor(
                target_size=self.config['image_processing']['target_size']
            )
            self.feature_detector = FeatureDetector()
            self.noise_reducer = NoiseReducer()
            
            # AI models
            self.segmentation_model = None
            self.depth_model = None
            self.room_classifier = None
            
            # Vectorization components
            self.vectorizer = Vectorizer(
                scale_factor=self.config['vectorization']['scale_factor']
            )
            self.geometry_processor = GeometryProcessor(
                tolerance=self.config['vectorization']['tolerance']
            )
            
            # 3D reconstruction
            self.model_builder = ModelBuilder(
                scale_factor=self.config['3d_reconstruction']['scale_factor']
            )
            
            self.logger.info("All pipeline components initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def _load_models(self):
        """Load pre-trained models if available."""
        try:
            model_config = self.config['models']
            device = model_config['device']
            
            # Load segmentation model
            seg_path = model_config['segmentation_model_path']
            if os.path.exists(seg_path):
                self.segmentation_model = FloorPlanSegmentationModel.load_model(seg_path, device)
                self.logger.info("Segmentation model loaded")
            else:
                self.logger.warning(f"Segmentation model not found at {seg_path}")
            
            # Load depth estimation model
            depth_path = model_config['depth_model_path']
            if os.path.exists(depth_path):
                self.depth_model = DepthEstimationModel.load_model(depth_path, device)
                self.logger.info("Depth estimation model loaded")
            else:
                self.logger.warning(f"Depth estimation model not found at {depth_path}")
            
            # Load room classifier
            room_path = model_config['room_classifier_path']
            if os.path.exists(room_path):
                self.room_classifier = RoomClassifier.load_model(room_path, device)
                self.logger.info("Room classifier loaded")
            else:
                self.logger.warning(f"Room classifier not found at {room_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            # Continue without models - will use fallback methods
    
    def convert_2d_to_3d(self, image_path: str, output_dir: str, 
                        output_name: str = "converted_model") -> Dict[str, Any]:
        """
        Convert a 2D floor plan to 3D model.
        
        Args:
            image_path: Path to the 2D floor plan image
            output_dir: Directory to save the 3D model
            output_name: Name for the output files
            
        Returns:
            Dictionary containing conversion results and metadata
        """
        try:
            self.logger.info(f"Starting 2D to 3D conversion for {image_path}")
            
            # Step 1: Image preprocessing
            self.logger.info("Step 1: Image preprocessing")
            processed_image = self._preprocess_image(image_path)
            
            # Step 2: Feature detection
            self.logger.info("Step 2: Feature detection")
            features = self._detect_features(processed_image)
            
            # Step 3: AI segmentation
            self.logger.info("Step 3: AI segmentation")
            segmentation_mask = self._segment_image(processed_image)
            
            # Step 4: Depth estimation
            self.logger.info("Step 4: Depth estimation")
            depth_predictions = self._estimate_depth(segmentation_mask)
            
            # Step 5: Room classification
            self.logger.info("Step 5: Room classification")
            room_classifications = self._classify_rooms(segmentation_mask, features)
            
            # Step 6: Vectorization
            self.logger.info("Step 6: Vectorization")
            vector_data = self._vectorize_data(segmentation_mask, features)
            
            # Step 7: Geometry processing
            self.logger.info("Step 7: Geometry processing")
            processed_vector_data = self._process_geometry(vector_data)
            
            # Step 8: 3D reconstruction
            self.logger.info("Step 8: 3D reconstruction")
            model_info = self._build_3d_model(processed_vector_data, depth_predictions)
            
            # Step 9: Export models
            self.logger.info("Step 9: Exporting models")
            export_paths = self._export_models(output_dir, output_name)
            
            # Step 10: Generate visualization
            self.logger.info("Step 10: Generating visualization")
            visualization_path = self._generate_visualization(
                processed_image, segmentation_mask, features, output_dir, output_name
            )
            
            # Compile results
            results = {
                'success': True,
                'input_image': image_path,
                'output_directory': output_dir,
                'exported_models': export_paths,
                'visualization': visualization_path,
                'model_info': model_info,
                'depth_predictions': depth_predictions,
                'room_classifications': room_classifications,
                'vector_data': processed_vector_data,
                'metadata': {
                    'total_processing_time': 0,  # Would be calculated in practice
                    'models_used': self._get_models_used(),
                    'processing_steps': 10
                }
            }
            
            self.logger.info("2D to 3D conversion completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in 2D to 3D conversion: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_image': image_path,
                'output_directory': output_dir
            }
    
    def _preprocess_image(self, image_path: str) -> Dict[str, Any]:
        """Preprocess the input image."""
        try:
            # Load image
            image = self.image_processor.load_image(image_path)
            
            # Apply preprocessing pipeline
            processed_data = self.image_processor.preprocess(image)
            
            # Apply noise reduction
            binary_image = self.noise_reducer.preprocess_pipeline(
                processed_data['grayscale']
            )
            
            processed_data['final_binary'] = binary_image
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            raise
    
    def _detect_features(self, processed_image: Dict[str, Any]) -> Dict[str, List]:
        """Detect architectural features in the image."""
        try:
            binary_image = processed_image['final_binary']
            features = self.feature_detector.detect_features(binary_image)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error detecting features: {e}")
            return {}
    
    def _segment_image(self, processed_image: Dict[str, Any]) -> np.ndarray:
        """Segment the image using AI model."""
        try:
            if self.segmentation_model is not None:
                # Use AI model for segmentation
                binary_image = processed_image['final_binary']
                segmentation_mask = self.segmentation_model.predict(binary_image)
            else:
                # Fallback to simple thresholding
                self.logger.warning("Using fallback segmentation method")
                binary_image = processed_image['final_binary']
                segmentation_mask = self._fallback_segmentation(binary_image)
            
            return segmentation_mask
            
        except Exception as e:
            self.logger.error(f"Error segmenting image: {e}")
            # Return empty segmentation mask
            return np.zeros(processed_image['final_binary'].shape, dtype=np.uint8)
    
    def _fallback_segmentation(self, binary_image: np.ndarray) -> np.ndarray:
        """Fallback segmentation method when AI model is not available."""
        try:
            # Simple segmentation based on connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_image, connectivity=8
            )
            
            # Create segmentation mask
            segmentation_mask = np.zeros_like(binary_image, dtype=np.uint8)
            
            # Classify components based on size and shape
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Simple classification based on area and aspect ratio
                if area > 10000:  # Large areas are rooms
                    segmentation_mask[labels == i] = 4  # living_room
                elif area > 1000:  # Medium areas are walls
                    segmentation_mask[labels == i] = 1  # wall
                elif area > 100:  # Small areas are doors/windows
                    if width > height * 2:  # Wide objects are doors
                        segmentation_mask[labels == i] = 2  # door
                    else:  # Tall objects are windows
                        segmentation_mask[labels == i] = 3  # window
                else:  # Very small areas are furniture
                    segmentation_mask[labels == i] = 9  # furniture
            
            return segmentation_mask
            
        except Exception as e:
            self.logger.error(f"Error in fallback segmentation: {e}")
            return np.zeros_like(binary_image, dtype=np.uint8)
    
    def _estimate_depth(self, segmentation_mask: np.ndarray) -> Dict[str, float]:
        """Estimate depth and height information."""
        try:
            if self.depth_model is not None:
                # Use AI model for depth estimation
                depth_predictions = self.depth_model.predict_depth(segmentation_mask)
            else:
                # Fallback to default values
                self.logger.warning("Using fallback depth estimation")
                depth_predictions = {
                    'depth': 5.0,
                    'wall_height': 2.5,
                    'spatial_features': [0.0] * 6
                }
            
            return depth_predictions
            
        except Exception as e:
            self.logger.error(f"Error estimating depth: {e}")
            return {
                'depth': 5.0,
                'wall_height': 2.5,
                'spatial_features': [0.0] * 6
            }
    
    def _classify_rooms(self, segmentation_mask: np.ndarray, 
                       features: Dict[str, List]) -> List[Dict[str, Any]]:
        """Classify rooms using AI model."""
        try:
            if self.room_classifier is not None and 'rooms' in features:
                # Use AI model for room classification
                room_classifications = self.room_classifier.classify_rooms(
                    segmentation_mask, 
                    [room.contour for room in features['rooms']]
                )
            else:
                # Fallback to simple classification
                self.logger.warning("Using fallback room classification")
                room_classifications = []
                if 'rooms' in features:
                    for room in features['rooms']:
                        classification = {
                            'room_type': room.room_type,
                            'room_type_confidence': 0.7,
                            'area': room.area,
                            'aspect_ratio': 0.5,
                            'furniture_density': 0.3,
                            'contour': room.contour,
                            'area_pixels': room.area
                        }
                        room_classifications.append(classification)
            
            return room_classifications
            
        except Exception as e:
            self.logger.error(f"Error classifying rooms: {e}")
            return []
    
    def _vectorize_data(self, segmentation_mask: np.ndarray, 
                       features: Dict[str, List]) -> Dict[str, Any]:
        """Convert segmentation and features to vector format."""
        try:
            vector_data = self.vectorizer.vectorize_segmentation(segmentation_mask, features)
            return vector_data
            
        except Exception as e:
            self.logger.error(f"Error vectorizing data: {e}")
            return {}
    
    def _process_geometry(self, vector_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and optimize vector geometry."""
        try:
            processed_data = self.geometry_processor.process_vector_data(vector_data)
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing geometry: {e}")
            return vector_data
    
    def _build_3d_model(self, vector_data: Dict[str, Any], 
                       depth_predictions: Dict[str, float]) -> Dict[str, Any]:
        """Build 3D model from vector data."""
        try:
            model_info = self.model_builder.build_3d_model(vector_data, depth_predictions)
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error building 3D model: {e}")
            return {}
    
    def _export_models(self, output_dir: str, output_name: str) -> Dict[str, str]:
        """Export 3D models in various formats."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            export_paths = {}
            
            formats = self.config['output']['formats']
            
            for format in formats:
                filepath = os.path.join(output_dir, f"{output_name}.{format}")
                self.model_builder.export_model(filepath, format)
                export_paths[format] = filepath
            
            return export_paths
            
        except Exception as e:
            self.logger.error(f"Error exporting models: {e}")
            return {}
    
    def _generate_visualization(self, processed_image: Dict[str, Any], 
                              segmentation_mask: np.ndarray, 
                              features: Dict[str, List], 
                              output_dir: str, 
                              output_name: str) -> str:
        """Generate visualization of the conversion process."""
        try:
            # Create visualization
            vis_image = self.feature_detector.visualize_features(
                processed_image['grayscale'], features
            )
            
            # Add segmentation overlay
            if self.segmentation_model is not None:
                seg_vis = self.segmentation_model.visualize_segmentation(
                    processed_image['grayscale'], segmentation_mask
                )
                vis_image = cv2.addWeighted(vis_image, 0.7, seg_vis, 0.3, 0)
            
            # Save visualization
            vis_path = os.path.join(output_dir, f"{output_name}_visualization.png")
            cv2.imwrite(vis_path, vis_image)
            
            return vis_path
            
        except Exception as e:
            self.logger.error(f"Error generating visualization: {e}")
            return ""
    
    def _get_models_used(self) -> List[str]:
        """Get list of models that were used in the conversion."""
        models_used = []
        
        if self.segmentation_model is not None:
            models_used.append("segmentation_model")
        if self.depth_model is not None:
            models_used.append("depth_model")
        if self.room_classifier is not None:
            models_used.append("room_classifier")
        
        return models_used
    
    def render_3d_model(self, output_dir: str, output_name: str, 
                       resolution: Tuple[int, int] = (1920, 1080)) -> str:
        """Render the 3D model to an image."""
        try:
            render_path = os.path.join(output_dir, f"{output_name}_render.png")
            self.model_builder.render_image(render_path, resolution)
            return render_path
            
        except Exception as e:
            self.logger.error(f"Error rendering 3D model: {e}")
            return ""
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'model_builder'):
                self.model_builder.cleanup()
            
            self.logger.info("AI Agent cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the AI Agent."""
        return {
            'models_loaded': {
                'segmentation_model': self.segmentation_model is not None,
                'depth_model': self.depth_model is not None,
                'room_classifier': self.room_classifier is not None
            },
            'config': self.config,
            'components_initialized': True
        }