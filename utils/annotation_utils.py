import os
import ast
import cv2
import numpy as np
import math
import sys
from PIL import Image, ImageDraw, ImageFont

# --- Gracefully import the emoji library ---
try:
    import emoji
    EMOJI_SUPPORTED = True
except ImportError:
    EMOJI_SUPPORTED = False
    print("Warning: 'emoji' library not found. Emoji icons will be replaced with text placeholders.")

def find_emoji_font(size=48):
    """Tries to load a common emoji font; falls back to the default font if not found."""
    CANDIDATE_FONTS = []
    if sys.platform.startswith("win"):
        CANDIDATE_FONTS += [r"C:\Windows\Fonts\seguiemj.ttf", r"C:\Windows\Fonts\seguihis.ttf"]
    elif sys.platform == "darwin":
        CANDIDATE_FONTS += ["/System/Library/Fonts/Apple Color Emoji.ttc", "/System/Library/Fonts/Supplemental/Apple Color Emoji.ttc"]
    else:
        CANDIDATE_FONTS += ["/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", "/usr/share/fonts/NotoColorEmoji.ttf", "/usr/local/share/fonts/NotoColorEmoji.ttf"]
    
    env_font = os.environ.get("EMOJI_FONT")
    if env_font:
        CANDIDATE_FONTS.insert(0, env_font)

    for path in CANDIDATE_FONTS:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

COLOR_MAP = {
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
}

def parse_visual_prompts(prompt_string):
    """
    Parses clean command strings from the 'visual prompts' key.
    Returns a list of extracted coordinate instructions.
    """
    commands = []
    if not prompt_string:
        return commands
        
    lines = prompt_string.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue

        try:
            command_name_str, dict_str = line.split(':', 1)
            command_name = command_name_str.strip().lower() 
            
            # Clean up potential trailing quotes
            dict_str = dict_str.strip()
            last_brace = dict_str.rfind('}')
            if last_brace != -1:
                dict_str = dict_str[:last_brace+1]
            
            params = ast.literal_eval(dict_str)

            if command_name == 'draw straight arrow':
                commands.append({
                    'type': 'move_arrow',
                    'start': tuple(params['start_point']), 
                    'end': tuple(params['end_point']),
                    'colors': [COLOR_MAP.get(c.lower(), (255, 255, 255)) for c in params.get('color', ['green'])]
                })
            
            elif command_name == 'draw rotating arrow':
                commands.append({
                    'type': 'rotate_arrow',
                    'center': tuple(params['center']), 
                    'direction': params.get('direction', 'clockwise').lower()
                })

            elif command_name == 'draw dual crosshair':
                commands.append({
                    'type': 'dual_crosshair',
                    'p1': tuple(params['start_point']), 
                    'p2': tuple(params['end_point'])
                })

            elif command_name in ['draw crosshair', 'draw on', 'draw off', 'draw lock', 'draw rewind']:
                pos = tuple(params['position']) 
                
                if command_name == 'draw crosshair':
                    commands.append({'type': 'crosshair', 'pos': pos})
                elif command_name == 'draw on':
                    commands.append({'type': 'state', 'pos': pos, 'text': 'ON'})
                elif command_name == 'draw off':
                    commands.append({'type': 'state', 'pos': pos, 'text': 'OFF'})
                elif command_name == 'draw lock':
                    commands.append({'type': 'lock', 'pos': pos})
                elif command_name == 'draw rewind':
                    commands.append({'type': 'rewind', 'pos': pos})
            else:
                print(f"Warning: Unknown visual prompt command: '{command_name_str}'")
                
        except (ValueError, SyntaxError, TypeError, KeyError) as e:
            print(f"Warning: Skipping command line (parse error: {e}). Line: '{line}'")

    return commands

class AnnotationRenderer:
    """A class to render annotations on an image using OpenCV."""
    
    def __init__(self, image, use_normalized_coordinates=True):
        """
        Args:
            image: OpenCV image array.
            use_normalized_coordinates: If True, scales coordinates from 0-1000 to image dimensions.
        """
        self.image = image
        self.use_normalized_coordinates = use_normalized_coordinates
        
        if self.image is not None:
            self.height, self.width, *_ = self.image.shape
        else:
            self.height, self.width = 0, 0 

    def _denormalize(self, point):
        """
        Converts 0-1000 normalized coordinates to absolute image coordinates if scaling is enabled.
        Otherwise, returns the original integer coordinates.
        """
        if not self.use_normalized_coordinates:
            return (int(point[0]), int(point[1]))

        if self.width == 0 or self.height == 0:
            return (0, 0)
            
        abs_x = int(point[0] / 1000.0 * self.width)
        abs_y = int(point[1] / 1000.0 * self.height)
        return (abs_x, abs_y)

    def draw_commands(self, commands):
        """Executes a list of drawing commands on the image."""
        if not isinstance(commands, list):
            print(f"Warning: 'commands' is not a list, skipping. Value: {commands}")
            return
            
        for command in commands:
            command_type = command.get('type')
            
            if command_type == 'move_arrow':
                self._draw_zebra_arrow(command['start'], command['end'], command.get('colors', ''))
            elif command_type == 'rotate_arrow':
                self._draw_rotation_arrow(command['center'], command.get('direction', ''))
            elif command_type == 'state':
                self._draw_state_marker(command['pos'], command.get('text', ''))
            elif command_type == 'lock':
                self._draw_lock_icon(command['pos'])
            elif command_type == 'rewind':
                self._draw_rewind_icon(command['pos'])
            elif command_type == 'crosshair':
                self._draw_crosshair(command['pos'])
            elif command_type == 'dual_crosshair':
                self._draw_dual_crosshair(command['p1'], command['p2'])
            else:
                print(f"Warning: Unknown command type '{command_type}'")

    def _draw_zebra_arrow(self, start_point_norm, end_point_norm, colors):
        start_point = self._denormalize(start_point_norm)
        end_point = self._denormalize(end_point_norm)
        
        if not colors: return
        outline_color, base_dim = (0,0,0), max(self.image.shape[0], self.image.shape[1])
        line_width = max(4, int(base_dim*0.008))
        outline_width = line_width+2
        head_length, head_width, segment_length = max(15, int(base_dim*0.025)), max(10, int(base_dim*0.015)), 10.0
        dx, dy, line_length = float(end_point[0]-start_point[0]), float(end_point[1]-start_point[1]), math.hypot(end_point[0]-start_point[0], end_point[1]-start_point[1])
        if line_length < head_length: return
        udx, udy = dx/line_length, dy/line_length
        line_actual_end = (int(end_point[0]-udx*head_length/2), int(end_point[1]-udy*head_length/2))
        cv2.line(self.image, start_point, line_actual_end, outline_color, thickness=outline_width, lineType=cv2.LINE_AA)
        line_body_length = math.hypot(line_actual_end[0]-start_point[0], line_actual_end[1]-start_point[1])
        num_segments = int(line_body_length/segment_length)
        for i in range(num_segments): 
            color = colors[i%len(colors)]
            seg_start = (int(start_point[0]+i*segment_length*udx), int(start_point[1]+i*segment_length*udy))
            seg_end = (int(start_point[0]+(i+1)*segment_length*udx), int(start_point[1]+(i+1)*segment_length*udy))
            cv2.line(self.image, seg_start, seg_end, color, thickness=line_width, lineType=cv2.LINE_AA)
        if line_body_length > num_segments*segment_length: 
            color = colors[num_segments%len(colors)]
            seg_start = (int(start_point[0]+num_segments*segment_length*udx), int(start_point[1]+num_segments*segment_length*udy))
            cv2.line(self.image, seg_start, line_actual_end, color, thickness=line_width, lineType=cv2.LINE_AA)
        p1 = (int(end_point[0]-udx*head_length-udy*head_width), int(end_point[1]-udy*head_length+udx*head_width))
        p2 = (int(end_point[0]-udx*head_length+udy*head_width), int(end_point[1]-udy*head_length-udx*head_width))
        arrowhead_pts = np.array([end_point, p1, p2], dtype=np.int32)
        final_segment_index = int((line_length-1)/segment_length)
        head_color = colors[final_segment_index%len(colors)]
        cv2.fillPoly(self.image, [arrowhead_pts], head_color, lineType=cv2.LINE_AA)
        cv2.polylines(self.image, [arrowhead_pts], isClosed=True, color=outline_color, thickness=1, lineType=cv2.LINE_AA)

    def _draw_rotation_arrow(self, center_norm, direction, radius=30, arc_degrees=270):
        center = self._denormalize(center_norm)
        is_clockwise = direction == 'clockwise'
        outline_color, body_color = (0, 0, 0), (255, 255, 255)
        start_angle_default = 0
        end_angle_default = arc_degrees
        if not is_clockwise:
            start_angle_default, end_angle_default = 360 - end_angle_default, 360
        outline_thickness, body_thickness = 4, 2
        try:
            cv2.ellipse(self.image, center, (radius, radius), 0, start_angle_default, end_angle_default, outline_color, outline_thickness, cv2.LINE_AA)
            cv2.ellipse(self.image, center, (radius, radius), 0, start_angle_default, end_angle_default, body_color, body_thickness, cv2.LINE_AA)
        except cv2.error as e:
            print(f"Error drawing ellipse with center {center} and radius {radius}: {e}")
            return
        end_rad = math.radians(end_angle_default if is_clockwise else start_angle_default)
        arrow_end = (int(center[0] + radius * math.cos(end_rad)), int(center[1] + radius * math.sin(end_rad)))
        arrow_length = max(10, int(radius * 0.4))
        arrow_spread = math.pi / 6
        arrow_angle = end_rad + (math.pi / 2 if is_clockwise else -math.pi / 2)
        arrow_line1_end = (int(arrow_end[0] - arrow_length * math.cos(arrow_angle - arrow_spread)), int(arrow_end[1] - arrow_length * math.sin(arrow_angle - arrow_spread)))
        arrow_line2_end = (int(arrow_end[0] - arrow_length * math.cos(arrow_angle + arrow_spread)), int(arrow_end[1] - arrow_length * math.sin(arrow_angle + arrow_spread)))
        cv2.line(self.image, arrow_end, arrow_line1_end, outline_color, outline_thickness, cv2.LINE_AA)
        cv2.line(self.image, arrow_end, arrow_line2_end, outline_color, outline_thickness, cv2.LINE_AA)
        cv2.line(self.image, arrow_end, arrow_line1_end, body_color, body_thickness, cv2.LINE_AA)
        cv2.line(self.image, arrow_end, arrow_line2_end, body_color, body_thickness, cv2.LINE_AA)

    def _draw_state_marker(self, position_norm, text):
        position = self._denormalize(position_norm)
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        outline_color, text_color = (0,0,0), (255,255,255)
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        text_origin = (position[0] - text_width // 2, position[1] + text_height // 2)
        cv2.putText(self.image, text, text_origin, font, scale, outline_color, thickness + 2, cv2.LINE_AA)
        cv2.putText(self.image, text, text_origin, font, scale, text_color, thickness, cv2.LINE_AA)

    def _draw_emoji(self, position_norm, shortcode, size=220):
        position = self._denormalize(position_norm)

        icon_dir = "./utils/emoji_icons"
        icon_path = os.path.join(icon_dir, shortcode.strip(":") + ".png")

        if os.path.exists(icon_path):
            try:
                icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    h, w = icon.shape[:2]
                    scale = size / max(h, w)
                    icon = cv2.resize(icon, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    ih, iw = icon.shape[:2]

                    x, y = position
                    y1, y2 = max(0, y - ih // 2), min(self.height, y + ih // 2)
                    x1, x2 = max(0, x - iw // 2), min(self.width, x + iw // 2)

                    icon_crop = icon[max(0, ih // 2 - y):(ih - max(0, y + ih // 2 - self.height)),
                                     max(0, iw // 2 - x):(iw - max(0, x + iw // 2 - self.width))]
                    bg_crop = self.image[y1:y2, x1:x2]
                    
                    if icon_crop.shape[:2] == bg_crop.shape[:2]:
                        alpha = icon_crop[..., 3:] / 255.0
                        self.image[y1:y2, x1:x2] = (1 - alpha) * bg_crop + alpha * icon_crop[..., :3]
                        return
            except Exception as e:
                print(f"Error drawing emoji from PNG ({shortcode}): {e}")

    def _draw_lock_icon(self, position_norm):
        self._draw_emoji(position_norm, shortcode=':prohibited:')

    def _draw_rewind_icon(self, position_norm):
        self._draw_emoji(position_norm, shortcode=':fast_reverse_button:')
        
    def _draw_crosshair(self, position_norm, size=30, gap=12, outline_thickness=4, fill_thickness=2):
        position = self._denormalize(position_norm)
        outline_color, fill_color = (0,0,0), (255,255,255)
        cx,cy = position; half_size,half_gap = size//2,gap//2
        points = [((cx-half_size,cy),(cx-half_gap,cy)),((cx+half_gap,cy),(cx+half_size,cy)),((cx,cy-half_size),(cx,cy-half_gap)),((cx,cy+half_gap),(cx,cy+half_size))]
        for p1,p2 in points: 
            cv2.line(self.image, p1, p2, outline_color, outline_thickness, cv2.LINE_AA)
            cv2.line(self.image, p1, p2, fill_color, fill_thickness, cv2.LINE_AA)

    def _draw_dashed_line(self, p1_norm, p2_norm, outline_thickness=4, fill_thickness=2, dash_len=10):
        p1 = self._denormalize(p1_norm)
        p2 = self._denormalize(p2_norm)
        outline_color, fill_color = (0,0,0), (255,255,255)
        dx,dy = p2[0]-p1[0], p2[1]-p1[1]; dist = math.hypot(dx,dy)
        if dist==0: return
        dashes = int(dist/dash_len)
        for i in range(dashes):
            if i%2==0:
                start_pos, end_pos = (int(p1[0]+(dx*i*dash_len)/dist),int(p1[1]+(dy*i*dash_len)/dist)), (int(p1[0]+(dx*(i+1)*dash_len)/dist),int(p1[1]+(dy*(i+1)*dash_len)/dist))
                cv2.line(self.image, start_pos, end_pos, outline_color, outline_thickness, cv2.LINE_AA)
                cv2.line(self.image, start_pos, end_pos, fill_color, fill_thickness, cv2.LINE_AA)

    def _draw_dual_crosshair(self, p1_norm, p2_norm): 
        self._draw_crosshair(p1_norm)
        self._draw_crosshair(p2_norm)
        self._draw_dashed_line(p1_norm, p2_norm)