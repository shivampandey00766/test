"""
SVG export module for vector data.

This module provides functionality to export vector data to SVG format
for use in CAD applications and web visualization.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Optional, Tuple
import numpy as np
from .raster_to_vector import VectorLine, VectorPolygon, VectorPoint, VectorData


class SVGExporter:
    """
    Exports vector data to SVG format.
    
    This class provides methods to convert vector data into
    well-formatted SVG files suitable for CAD applications.
    """
    
    def __init__(self, 
                 stroke_width: float = 2.0,
                 fill_opacity: float = 0.3,
                 stroke_opacity: float = 1.0):
        """
        Initialize the SVGExporter.
        
        Args:
            stroke_width: Default stroke width for lines
            fill_opacity: Default fill opacity for polygons
            stroke_opacity: Default stroke opacity for lines
        """
        self.stroke_width = stroke_width
        self.fill_opacity = fill_opacity
        self.stroke_opacity = stroke_opacity
        
        # Color scheme for different element types
        self.colors = {
            'wall': '#000000',      # Black
            'door': '#00FF00',      # Green
            'window': '#0000FF',    # Blue
            'room': '#FF0000',      # Red
            'default': '#808080'    # Gray
        }
    
    def export_to_svg(self, vector_data: VectorData, 
                     output_path: str,
                     width: Optional[float] = None,
                     height: Optional[float] = None,
                     viewbox: Optional[Tuple[float, float, float, float]] = None) -> None:
        """
        Export vector data to SVG file.
        
        Args:
            vector_data: VectorData object
            output_path: Output file path
            width: SVG width (optional)
            height: SVG height (optional)
            viewbox: SVG viewbox (x, y, width, height)
        """
        # Calculate dimensions if not provided
        if width is None or height is None or viewbox is None:
            width, height, viewbox = self._calculate_dimensions(vector_data)
        
        # Create SVG root element
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('viewBox', f"{viewbox[0]} {viewbox[1]} {viewbox[2]} {viewbox[3]}")
        
        # Add style definitions
        self._add_styles(svg)
        
        # Add polygons first (so they appear behind lines)
        for polygon in vector_data.polygons:
            self._add_polygon(svg, polygon)
        
        # Add lines
        for line in vector_data.lines:
            self._add_line(svg, line)
        
        # Add points
        for point in vector_data.points:
            self._add_point(svg, point)
        
        # Add metadata as comments
        self._add_metadata(svg, vector_data)
        
        # Format and save
        self._save_svg(svg, output_path)
    
    def _calculate_dimensions(self, vector_data: VectorData) -> Tuple[float, float, Tuple[float, float, float, float]]:
        """Calculate SVG dimensions and viewbox."""
        all_x = []
        all_y = []
        
        # Collect all coordinates
        for line in vector_data.lines:
            all_x.extend([line.start.x, line.end.x])
            all_y.extend([line.start.y, line.end.y])
        
        for polygon in vector_data.polygons:
            for point in polygon.points:
                all_x.append(point.x)
                all_y.append(point.y)
        
        for point in vector_data.points:
            all_x.append(point.x)
            all_y.append(point.y)
        
        if not all_x or not all_y:
            return 800, 600, (0, 0, 800, 600)
        
        # Calculate bounding box
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Add padding
        padding = 50
        width = max_x - min_x + 2 * padding
        height = max_y - min_y + 2 * padding
        
        viewbox = (min_x - padding, min_y - padding, width, height)
        
        return width, height, viewbox
    
    def _add_styles(self, svg: ET.Element) -> None:
        """Add CSS styles to SVG."""
        style = ET.SubElement(svg, 'style')
        style.text = """
        .wall { stroke: #000000; stroke-width: 3; fill: none; }
        .door { stroke: #00FF00; stroke-width: 2; fill: none; }
        .window { stroke: #0000FF; stroke-width: 2; fill: none; }
        .room { stroke: #FF0000; stroke-width: 1; fill: #FF0000; fill-opacity: 0.1; }
        .point { fill: #808080; stroke: none; }
        """
    
    def _add_polygon(self, svg: ET.Element, polygon: VectorPolygon) -> None:
        """Add polygon to SVG."""
        if len(polygon.points) < 3:
            return
        
        # Create path data
        path_data = f"M {polygon.points[0].x} {polygon.points[0].y}"
        for point in polygon.points[1:]:
            path_data += f" L {point.x} {point.y}"
        path_data += " Z"  # Close path
        
        # Create path element
        path = ET.SubElement(svg, 'path')
        path.set('d', path_data)
        path.set('class', polygon.polygon_type)
        
        # Add title for tooltip
        title = ET.SubElement(path, 'title')
        title.text = f"{polygon.polygon_type} (area: {polygon.area:.1f})"
    
    def _add_line(self, svg: ET.Element, line: VectorLine) -> None:
        """Add line to SVG."""
        line_elem = ET.SubElement(svg, 'line')
        line_elem.set('x1', str(line.start.x))
        line_elem.set('y1', str(line.start.y))
        line_elem.set('x2', str(line.end.x))
        line_elem.set('y2', str(line.end.y))
        line_elem.set('class', line.line_type)
        line_elem.set('stroke-width', str(max(1, line.thickness)))
        
        # Add title for tooltip
        length = np.sqrt((line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2)
        title = ET.SubElement(line_elem, 'title')
        title.text = f"{line.line_type} (length: {length:.1f})"
    
    def _add_point(self, svg: ET.Element, point: VectorPoint) -> None:
        """Add point to SVG."""
        circle = ET.SubElement(svg, 'circle')
        circle.set('cx', str(point.x))
        circle.set('cy', str(point.y))
        circle.set('r', '3')
        circle.set('class', 'point')
        
        # Add title for tooltip
        title = ET.SubElement(circle, 'title')
        title.text = f"Point ({point.x:.1f}, {point.y:.1f})"
    
    def _add_metadata(self, svg: ET.Element, vector_data: VectorData) -> None:
        """Add metadata as comments."""
        metadata = ET.Comment(f"""
        Generated by AI Agent for 2D to 3D Architectural Conversion
        Timestamp: {vector_data.metadata.get('conversion_timestamp', 'unknown')}
        Image Shape: {vector_data.metadata.get('image_shape', 'unknown')}
        Lines: {len(vector_data.lines)}
        Polygons: {len(vector_data.polygons)}
        Points: {len(vector_data.points)}
        """)
        svg.insert(0, metadata)
    
    def _save_svg(self, svg: ET.Element, output_path: str) -> None:
        """Save SVG to file with proper formatting."""
        # Convert to string
        rough_string = ET.tostring(svg, 'unicode')
        
        # Parse and format
        reparsed = minidom.parseString(rough_string)
        formatted = reparsed.toprettyxml(indent="  ")
        
        # Remove empty lines
        lines = [line for line in formatted.split('\n') if line.strip()]
        formatted = '\n'.join(lines)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted)
    
    def export_layered_svg(self, vector_data: VectorData, 
                          output_path: str,
                          separate_layers: bool = True) -> None:
        """
        Export SVG with separate layers for different element types.
        
        Args:
            vector_data: VectorData object
            output_path: Output file path
            separate_layers: Whether to create separate layers
        """
        if not separate_layers:
            self.export_to_svg(vector_data, output_path)
            return
        
        # Calculate dimensions
        width, height, viewbox = self._calculate_dimensions(vector_data)
        
        # Create SVG root element
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('viewBox', f"{viewbox[0]} {viewbox[1]} {viewbox[2]} {viewbox[3]}")
        
        # Add style definitions
        self._add_styles(svg)
        
        # Group elements by type
        element_groups = {
            'rooms': [],
            'walls': [],
            'doors': [],
            'windows': [],
            'points': []
        }
        
        for polygon in vector_data.polygons:
            if polygon.polygon_type == 'room':
                element_groups['rooms'].append(polygon)
            elif polygon.polygon_type == 'door':
                element_groups['doors'].append(polygon)
            elif polygon.polygon_type == 'window':
                element_groups['windows'].append(polygon)
        
        for line in vector_data.lines:
            if line.line_type == 'wall':
                element_groups['walls'].append(line)
            elif line.line_type == 'door':
                element_groups['doors'].append(line)
            elif line.line_type == 'window':
                element_groups['windows'].append(line)
        
        element_groups['points'] = vector_data.points
        
        # Create layers
        for layer_name, elements in element_groups.items():
            if not elements:
                continue
            
            layer = ET.SubElement(svg, 'g')
            layer.set('id', layer_name)
            layer.set('class', 'layer')
            
            # Add elements to layer
            if layer_name == 'rooms':
                for polygon in elements:
                    self._add_polygon(layer, polygon)
            elif layer_name in ['doors', 'windows']:
                for polygon in elements:
                    self._add_polygon(layer, polygon)
                for line in elements:
                    self._add_line(layer, line)
            elif layer_name == 'walls':
                for line in elements:
                    self._add_line(layer, line)
            elif layer_name == 'points':
                for point in elements:
                    self._add_point(layer, point)
        
        # Add metadata
        self._add_metadata(svg, vector_data)
        
        # Save
        self._save_svg(svg, output_path)
    
    def create_interactive_svg(self, vector_data: VectorData, 
                              output_path: str) -> None:
        """
        Create interactive SVG with JavaScript functionality.
        
        Args:
            vector_data: VectorData object
            output_path: Output file path
        """
        # Calculate dimensions
        width, height, viewbox = self._calculate_dimensions(vector_data)
        
        # Create SVG root element
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('viewBox', f"{viewbox[0]} {viewbox[1]} {viewbox[2]} {viewbox[3]}")
        
        # Add JavaScript for interactivity
        script = ET.SubElement(svg, 'script')
        script.set('type', 'text/javascript')
        script.text = """
        function highlightElement(element) {
            element.style.stroke = '#FFD700';
            element.style.strokeWidth = '4';
        }
        
        function unhighlightElement(element) {
            element.style.stroke = '';
            element.style.strokeWidth = '';
        }
        
        function toggleLayer(layerId) {
            var layer = document.getElementById(layerId);
            if (layer.style.display === 'none') {
                layer.style.display = 'block';
            } else {
                layer.style.display = 'none';
            }
        }
        """
        
        # Add style definitions
        self._add_styles(svg)
        
        # Add interactive styles
        interactive_style = ET.SubElement(svg, 'style')
        interactive_style.text = """
        .interactive:hover { stroke: #FFD700; stroke-width: 4; cursor: pointer; }
        .layer-control { position: absolute; top: 10px; left: 10px; background: white; padding: 10px; border: 1px solid #ccc; }
        """
        
        # Add layer controls
        controls = ET.SubElement(svg, 'foreignObject')
        controls.set('x', '10')
        controls.set('y', '10')
        controls.set('width', '200')
        controls.set('height', '100')
        
        div = ET.SubElement(controls, 'div')
        div.set('class', 'layer-control')
        
        # Add control buttons
        for layer_name in ['rooms', 'walls', 'doors', 'windows']:
            button = ET.SubElement(div, 'button')
            button.set('onclick', f'toggleLayer("{layer_name}")')
            button.text = f'Toggle {layer_name.title()}'
        
        # Add elements with interactivity
        for polygon in vector_data.polygons:
            if len(polygon.points) >= 3:
                path_data = f"M {polygon.points[0].x} {polygon.points[0].y}"
                for point in polygon.points[1:]:
                    path_data += f" L {point.x} {point.y}"
                path_data += " Z"
                
                path = ET.SubElement(svg, 'path')
                path.set('d', path_data)
                path.set('class', f"{polygon.polygon_type} interactive")
                path.set('onmouseover', 'highlightElement(this)')
                path.set('onmouseout', 'unhighlightElement(this)')
        
        for line in vector_data.lines:
            line_elem = ET.SubElement(svg, 'line')
            line_elem.set('x1', str(line.start.x))
            line_elem.set('y1', str(line.start.y))
            line_elem.set('x2', str(line.end.x))
            line_elem.set('y2', str(line.end.y))
            line_elem.set('class', f"{line.line_type} interactive")
            line_elem.set('onmouseover', 'highlightElement(this)')
            line_elem.set('onmouseout', 'unhighlightElement(this)')
        
        # Save
        self._save_svg(svg, output_path)
    
    def export_to_dxf(self, vector_data: VectorData, output_path: str) -> None:
        """
        Export vector data to DXF format (simplified).
        
        Note: This is a basic implementation. For production use,
        consider using a dedicated DXF library like ezdxf.
        
        Args:
            vector_data: VectorData object
            output_path: Output file path
        """
        # This is a simplified DXF export
        # For production use, use a proper DXF library
        
        dxf_content = [
            "0",
            "SECTION",
            "2",
            "ENTITIES",
        ]
        
        # Add lines
        for line in vector_data.lines:
            dxf_content.extend([
                "0",
                "LINE",
                "8",  # Layer
                line.line_type.upper(),
                "10",  # Start X
                str(line.start.x),
                "20",  # Start Y
                str(line.start.y),
                "30",  # Start Z
                str(line.start.z),
                "11",  # End X
                str(line.end.x),
                "21",  # End Y
                str(line.end.y),
                "31",  # End Z
                str(line.end.z),
            ])
        
        # Add polygons as polylines
        for polygon in vector_data.polygons:
            if len(polygon.points) >= 3:
                dxf_content.extend([
                    "0",
                    "LWPOLYLINE",
                    "8",  # Layer
                    polygon.polygon_type.upper(),
                    "90",  # Number of vertices
                    str(len(polygon.points)),
                    "70",  # Flags (1 = closed)
                    "1",
                ])
                
                # Add vertices
                for i, point in enumerate(polygon.points):
                    dxf_content.extend([
                        "10",  # X
                        str(point.x),
                        "20",  # Y
                        str(point.y),
                    ])
        
        dxf_content.extend([
            "0",
            "ENDSEC",
            "0",
            "EOF"
        ])
        
        # Save DXF file
        with open(output_path, 'w') as f:
            f.write('\n'.join(dxf_content))