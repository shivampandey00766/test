"""
Visualization Utilities

Provides visualization tools for debugging and result presentation.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Visualization utilities for the architectural AI agent.
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figure_size: Default figure size for matplotlib plots
        """
        self.figure_size = figure_size
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
    
    def visualize_preprocessing_results(self, original_image: np.ndarray, 
                                      processed_image: np.ndarray, 
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize image preprocessing results.
        
        Args:
            original_image: Original input image
            processed_image: Processed image
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figure_size)
        
        # Original image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Processed image
        axes[1].imshow(processed_image, cmap='gray')
        axes[1].set_title('Processed Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_feature_detection(self, image: np.ndarray, features_data: Dict,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize detected features on the image.
        
        Args:
            image: Input image
            features_data: Detected features data
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)
        
        # Display base image
        ax.imshow(image, cmap='gray', alpha=0.7)
        
        # Draw walls
        walls = features_data.get('walls', [])
        for wall in walls:
            if 'start' in wall and 'end' in wall:
                start, end = wall['start'], wall['end']
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       'r-', linewidth=3, label='Wall' if walls.index(wall) == 0 else '')
        
        # Draw doors
        doors = features_data.get('doors', [])
        for door in doors:
            if 'center' in door:
                center = door['center']
                radius = door.get('radius', 10)
                circle = plt.Circle(center, radius, fill=False, color='blue', linewidth=2)
                ax.add_patch(circle)
                if doors.index(door) == 0:
                    ax.plot([], [], 'bo', label='Door')
        
        # Draw windows
        windows = features_data.get('windows', [])
        for window in windows:
            if 'center' in window:
                center = window['center']
                width = window.get('width', 20)
                height = window.get('height', 10)
                rect = plt.Rectangle((center[0] - width/2, center[1] - height/2), 
                                   width, height, fill=False, color='green', linewidth=2)
                ax.add_patch(rect)
                if windows.index(window) == 0:
                    ax.plot([], [], 'gs', label='Window')
        
        # Draw fixtures
        fixtures = features_data.get('fixtures', [])
        for fixture in fixtures:
            if 'center' in fixture:
                center = fixture['center']
                ax.plot(center[0], center[1], 'mo', markersize=8)
                if fixtures.index(fixture) == 0:
                    ax.plot([], [], 'mo', label='Fixture')
        
        ax.set_title('Detected Features')
        ax.legend()
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_segmentation(self, image: np.ndarray, segmentation_map: np.ndarray,
                             class_names: Dict[int, str], save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize semantic segmentation results.
        
        Args:
            image: Original image
            segmentation_map: Segmentation result
            class_names: Mapping of class IDs to names
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation map
        unique_classes = np.unique(segmentation_map)
        cmap = plt.cm.get_cmap('tab20', len(unique_classes))
        
        axes[1].imshow(segmentation_map, cmap=cmap)
        axes[1].set_title('Segmentation Map')
        axes[1].axis('off')
        
        # Overlay
        overlay = np.zeros((*segmentation_map.shape, 3))
        for i, class_id in enumerate(unique_classes):
            mask = segmentation_map == class_id
            color = np.array(cmap(i)[:3])
            overlay[mask] = color
        
        axes[2].imshow(image, cmap='gray', alpha=0.7)
        axes[2].imshow(overlay, alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # Create legend
        legend_elements = []
        for class_id in unique_classes:
            if class_id in class_names:
                color = cmap(list(unique_classes).index(class_id))
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                                   label=class_names[class_id]))
        
        if legend_elements:
            axes[2].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_depth_estimation(self, image: np.ndarray, depth_map: np.ndarray,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize depth estimation results.
        
        Args:
            image: Original image
            depth_map: Estimated depth map
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Depth map
        im1 = axes[1].imshow(depth_map, cmap='viridis')
        axes[1].set_title('Depth Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3D surface plot
        y, x = np.mgrid[0:depth_map.shape[0]:10, 0:depth_map.shape[1]:10]
        z = depth_map[::10, ::10]
        
        axes[2].remove()  # Remove the 2D axis
        ax3d = fig.add_subplot(133, projection='3d')
        ax3d.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
        ax3d.set_title('3D Surface')
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Depth')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_3d_model_interactive(self, model_data: Dict) -> go.Figure:
        """
        Create interactive 3D visualization using Plotly.
        
        Args:
            model_data: 3D model data from reconstruction
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        scene = model_data.get('scene')
        if not scene:
            logger.warning("No scene data available for visualization")
            return fig
        
        # Add meshes to the plot
        for name, mesh in scene.geometry.items():
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                vertices = mesh.vertices
                faces = mesh.faces
                
                # Get color based on mesh name
                color = self._get_mesh_color(name)
                
                # Add mesh to plot
                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    name=name,
                    opacity=0.8 if 'window' in name.lower() else 1.0
                ))
        
        # Update layout
        fig.update_layout(
            title='3D Architectural Model',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def visualize_room_analysis(self, room_analysis: Dict) -> go.Figure:
        """
        Visualize room analysis results.
        
        Args:
            room_analysis: Room analysis data from segmentation
            
        Returns:
            Plotly figure
        """
        # Prepare data for visualization
        room_types = []
        room_counts = []
        room_areas = []
        
        for room_type, room_info in room_analysis.items():
            room_types.append(room_type.replace('room_', '').title())
            room_counts.append(room_info['count'])
            room_areas.append(room_info['total_area'])
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Room Count Distribution', 'Room Area Distribution'],
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Room count bar chart
        fig.add_trace(
            go.Bar(x=room_types, y=room_counts, name='Count', 
                  marker_color=self.color_palette[:len(room_types)]),
            row=1, col=1
        )
        
        # Room area bar chart
        fig.add_trace(
            go.Bar(x=room_types, y=room_areas, name='Area (px²)', 
                  marker_color=self.color_palette[:len(room_types)]),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Room Analysis Results',
            showlegend=False,
            height=500
        )
        
        return fig
    
    def create_processing_pipeline_visualization(self, pipeline_data: Dict) -> go.Figure:
        """
        Create a visualization of the processing pipeline stages.
        
        Args:
            pipeline_data: Data from each pipeline stage
            
        Returns:
            Plotly figure
        """
        # This would create a comprehensive visualization showing
        # the progression through each stage of the pipeline
        
        stages = list(pipeline_data.keys())
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=stages[:6],  # Show up to 6 stages
            specs=[[{'type': 'image'} for _ in range(3)] for _ in range(2)]
        )
        
        for i, (stage, data) in enumerate(pipeline_data.items()):
            if i >= 6:  # Limit to 6 subplots
                break
                
            row = i // 3 + 1
            col = i % 3 + 1
            
            # Add image or other visualization based on data type
            if isinstance(data, np.ndarray) and len(data.shape) == 2:
                fig.add_trace(
                    go.Heatmap(z=data, showscale=False, colorscale='gray'),
                    row=row, col=col
                )
        
        fig.update_layout(
            title='Processing Pipeline Visualization',
            height=800
        )
        
        return fig
    
    def _get_mesh_color(self, mesh_name: str) -> str:
        """Get color for mesh based on its name."""
        color_map = {
            'wall': 'lightgray',
            'floor': 'burlywood',
            'ceiling': 'white',
            'door': 'brown',
            'window': 'lightblue',
            'fixture': 'silver',
            'stairs': 'gray'
        }
        
        # Find matching color based on mesh name
        for key, color in color_map.items():
            if key in mesh_name.lower():
                return color
        
        return 'gray'  # Default color
    
    def save_interactive_plot(self, fig: go.Figure, filepath: str):
        """
        Save interactive Plotly figure.
        
        Args:
            fig: Plotly figure
            filepath: Output file path
        """
        try:
            if filepath.endswith('.html'):
                fig.write_html(filepath)
            elif filepath.endswith('.png'):
                fig.write_image(filepath)
            elif filepath.endswith('.pdf'):
                fig.write_image(filepath)
            else:
                logger.warning(f"Unsupported file format: {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
    
    def create_summary_report(self, results: Dict) -> str:
        """
        Create a text summary report of the analysis results.
        
        Args:
            results: Complete analysis results
            
        Returns:
            Summary report as string
        """
        report = []
        report.append("ARCHITECTURAL AI AGENT - ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Model metadata
        if 'metadata' in results:
            metadata = results['metadata']
            report.append("MODEL INFORMATION:")
            report.append(f"  Scale Factor: {metadata.get('scale_factor', 'N/A')}")
            report.append(f"  Units: {metadata.get('units', 'N/A')}")
            report.append(f"  Creation Time: {metadata.get('creation_timestamp', 'N/A')}")
            report.append("")
            
            # Model statistics
            model_info = metadata.get('model_info', {})
            report.append("DETECTED ELEMENTS:")
            report.append(f"  Rooms: {model_info.get('total_rooms', 0)}")
            report.append(f"  Walls: {model_info.get('total_walls', 0)}")
            report.append(f"  Doors: {model_info.get('total_doors', 0)}")
            report.append(f"  Windows: {model_info.get('total_windows', 0)}")
            report.append(f"  Fixtures: {model_info.get('total_fixtures', 0)}")
            report.append("")
            
            # Dimensions
            dimensions = metadata.get('dimensions', {})
            if 'floor_area' in dimensions:
                report.append(f"TOTAL FLOOR AREA: {dimensions['floor_area']:.2f} m²")
                report.append("")
        
        # Room analysis
        if 'room_analysis' in results:
            room_analysis = results['room_analysis']
            report.append("ROOM BREAKDOWN:")
            for room_type, room_info in room_analysis.items():
                room_name = room_type.replace('room_', '').title()
                report.append(f"  {room_name}: {room_info['count']} room(s)")
            report.append("")
        
        # 3D model statistics
        if 'total_vertices' in results:
            report.append("3D MODEL STATISTICS:")
            report.append(f"  Total Vertices: {results['total_vertices']:,}")
            report.append(f"  Total Faces: {results['total_faces']:,}")
            report.append("")
        
        return "\n".join(report)