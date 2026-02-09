import sys
import json
import socket
import math
import time
import os
import logging
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QSurfaceFormat, QFont, QColor, QPainter, QIcon, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QGroupBox, QGridLayout, QSlider,
    QTextEdit, QPushButton, QHeaderView, QSpacerItem, QSizePolicy, QFrame
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import pyqtgraph.opengl as gl  # Ensure pyqtgraph.opengl is imported
from OpenGL.GL import *
from OpenGL.GLU import *
import trimesh
from PIL import Image

# ================================
# Logging Configuration
# ================================
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# ================================
# Network Communication
# ================================

class FrontendClient:
    """
    Frontend client class that communicates with the backend via socket.
    Assumes that each JSON message from the backend ends with a newline character '\n'.
    """
    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self.ally_positions = []
        self.enemy_positions = []
        self.enemy_groups = []  # 新增属性，用于存储敌方组别及其威胁值
        self.human_preference = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]  # Initialize as floats

        self.connect_server()

    def connect_server(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((self.host, self.port))
            self.connected = True
            self.sock.settimeout(None)
            logging.info("Connected to backend")
        except Exception as e:
            logging.error(f"Failed to connect to backend: {e}")
            self.connected = False

    def close(self):
        if self.sock:
            self.sock.close()
        self.connected = False
        logging.info("Disconnected from backend")

    def send_preference(self, new_pref):
        if not self.connected:
            logging.warning("Not connected to backend. Cannot send preferences.")
            return
        data = {
            "type": "update_preference",
            "data": new_pref
        }
        msg = json.dumps(data) + "\n"  # Each JSON message ends with a newline
        try:
            self.sock.sendall(msg.encode('utf-8'))
            logging.info(f"Sent preferences: {new_pref}")
        except Exception as e:
            logging.error(f"Error sending preferences: {e}")
            self.connected = False

    def update_positions(self, info):
        self.ally_positions = info.get("ally", [])
        self.enemy_positions = info.get("enemy", [])
        self.enemy_groups = info.get("enemy_groups", [])  # 更新敌方组别信息
        self.human_preference = info.get("human_preference", self.human_preference)


class DataListenerThread(QThread):
    """
    A separate thread that continuously receives data using blocking recv from the socket.
    Each data message ends with a newline character, so the buffer is split by lines and JSON is parsed line by line.
    When data is received, it emits the data_received signal to send data to the main thread.
    """
    data_received = pyqtSignal(dict)

    def __init__(self, client: FrontendClient, parent=None):
        super().__init__(parent)
        self.client = client
        self.running = True
        self.buffer = ""

    def run(self):
        if not self.client.connected or self.client.sock is None:
            logging.warning("Not connected to backend. Data listener thread terminated.")
            return
        sock = self.client.sock
        sock.setblocking(True)  # Blocking mode
        logging.info("Data listener thread started.")
        while self.running:
            try:
                data = sock.recv(4096)
                if not data:
                    logging.warning("Backend has disconnected")
                    self.running = False
                    break
                self.buffer += data.decode('utf-8')
                self.parse_buffer_by_line()
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
                self.running = False
                break

    def parse_buffer_by_line(self):
        lines = self.buffer.split('\n')
        self.buffer = lines[-1]  # Keep the last incomplete line
        lines = lines[:-1]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                info = json.loads(line)
                self.data_received.emit(info)
                # logging.debug(f"Received data: {info}")
            except json.JSONDecodeError as ex:
                logging.error(f"JSON parsing failed: {ex}")

    def stop(self):
        self.running = False
        if self.client.connected:
            self.client.close()
        logging.info("Data listener thread stopped.")


# ================================
# 3D Rendering
# ================================

class Plane3D:
    """
    Represents a 3D airplane model, including model, texture, etc.
    Uses trimesh to load the model and convert it into OpenGL-renderable data.
    """
    def __init__(self, model_path, color=QColor("blue")):
        self.scene = None
        self.color = color
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.indices = []
        self.texture_id = None
        try:
            # Use absolute path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_model_path = os.path.join(current_dir, model_path)
            if not os.path.exists(full_model_path):
                raise FileNotFoundError(f"Model file not found: {full_model_path}")
            self.load_model(full_model_path)
            logging.info(f"Successfully loaded model: {full_model_path}")
        except Exception as e:
            logging.error(f"Failed to load model ({model_path}): {e}")

    def load_model(self, model_path):
        mesh = trimesh.load(model_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Cannot load model: {model_path}")

        # Get vertices, normals, and texture coordinates
        self.vertices = mesh.vertices.astype(np.float32)
        self.normals = mesh.vertex_normals.astype(np.float32) if mesh.vertex_normals is not None else None

        # Check for texture coordinates
        if mesh.visual.kind == 'texture' and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            self.texcoords = mesh.visual.uv.astype(np.float32)
        else:
            self.texcoords = None  # No texture coordinates

        self.indices = mesh.faces.flatten().astype(np.uint32)

        # Load texture
        if mesh.visual.kind == 'texture' and hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
            image = mesh.visual.material.image
            image = Image.fromarray(image)
            self.texture_id = self.load_texture(image)

    def load_texture(self, image):
        try:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = image.convert("RGBA").tobytes()
            width, height = image.size

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            logging.info(f"Texture loaded successfully: {image}")
            return texture_id
        except Exception as e:
            logging.error(f"Failed to load texture: {e}")
            return None

    def setup_buffers(self):
        """
        Set up VBOs
        """
        self.vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        if self.normals is not None:
            self.vbo_normals = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
            glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)
        else:
            self.vbo_normals = None

        if self.texcoords is not None:
            self.vbo_texcoords = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_texcoords)
            glBufferData(GL_ARRAY_BUFFER, self.texcoords.nbytes, self.texcoords, GL_STATIC_DRAW)
        else:
            self.vbo_texcoords = None

        self.vbo_indices = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

    def render(self):
        try:
            glPushMatrix()  # Start applying transformations
            glRotatef(180, 0, 1, 0)  # Rotate 180 degrees to align +Z if model defaults to -Z

            if not hasattr(self, 'vbo_vertices'):
                self.setup_buffers()

            glEnableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
            glVertexPointer(3, GL_FLOAT, 0, None)

            if self.normals is not None and hasattr(self, 'vbo_normals'):
                glEnableClientState(GL_NORMAL_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
                glNormalPointer(GL_FLOAT, 0, None)
            else:
                glDisableClientState(GL_NORMAL_ARRAY)

            if self.texcoords is not None and hasattr(self, 'vbo_texcoords'):
                glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo_texcoords)
                glTexCoordPointer(2, GL_FLOAT, 0, None)
            else:
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)

            if self.texture_id and self.texcoords is not None:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, self.texture_id)
            else:
                glDisable(GL_TEXTURE_2D)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_indices)
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

            glDisableClientState(GL_VERTEX_ARRAY)
            if self.normals is not None:
                glDisableClientState(GL_NORMAL_ARRAY)
            if self.texcoords is not None:
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)

            if self.texture_id and self.texcoords is not None:
                glDisable(GL_TEXTURE_2D)

        except Exception as e:
            logging.error(f"Exception during model rendering: {e}")
        finally:
            glPopMatrix()  # Ensure the stack is correctly released regardless of exceptions


class Plane3DInstance:
    """
    Manages a single airplane instance, including position, rotation, color, and speed.
    """
    def __init__(self, model: Plane3D, position, rotation, color=QColor("blue"), group=None, is_enemy=False):
        self.model = model  # Plane3D object
        self.position = position  # [x, y, z]
        self.rotation = rotation  # {"pitch":..., "yaw":..., "roll":...}
        self.color = color
        self.scale = 60  # Added scaling factor
        self.group = group  # Only enemies have group labels
        self.is_enemy = is_enemy  # Flag to indicate if it's an enemy plane
        self.command = [-1, -1]  # Initialize commands as [-1, -1] indicating no command
        self.speed = 0.0  # 新增属性，用于存储飞机速度

    def update_transform(self, position, rotation, color, group=None, command=None, speed=0.0):
        self.position = position
        self.rotation = rotation
        self.color = color
        if group is not None:
            self.group = group

        if command:
            self.command = command  # Update command information

        self.speed = speed  # 更新速度

    def render(self):
        try:
            glPushMatrix()
            glTranslatef(*self.position)

            # Apply rotations: yaw first, then pitch, then roll
            glRotatef(self.rotation.get("yaw", 0), 0, 1, 0)
            glRotatef(self.rotation.get("pitch", 0), 1, 0, 0)
            glRotatef(self.rotation.get("roll", 0), 0, 0, 1)

            # Apply scaling
            glScalef(self.scale, self.scale, self.scale)

            # Set color
            glColor3f(self.color.redF(), self.color.greenF(), self.color.blueF())
            self.model.render()
            glPopMatrix()
        except Exception as e:
            logging.error(f"Exception during plane instance rendering: {e}")


class Arrow3D:
    """
    Represents a 3D arrow used to indicate velocity vectors.
    Implemented as a line and a cone.
    """
    def __init__(self, start_pos, direction, length=50, color=QColor("red")):
        self.start_pos = start_pos  # [x, y, z]
        self.direction = direction  # [dx, dy, dz], normalized
        self.length = length
        self.color = color

    def render(self):
        try:
            end_pos = [
                self.start_pos[i] + self.direction[i] * self.length
                for i in range(3)
            ]
            glColor3f(self.color.redF(), self.color.greenF(), self.color.blueF())
            glLineWidth(2.0)
            glBegin(GL_LINES)
            glVertex3f(*self.start_pos)
            glVertex3f(*end_pos)
            glEnd()

            # Draw the arrowhead (cone)
            glPushMatrix()
            glTranslatef(*end_pos)
            # Calculate rotation angle
            norm = math.sqrt(sum([c**2 for c in self.direction]))
            if norm != 0:
                angle = math.degrees(math.acos(self.direction[2] / norm)) if norm != 0 else 0
                axis = [-self.direction[1], self.direction[0], 0]
                if any(axis):
                    glRotatef(angle, *axis)
            glColor3f(self.color.redF(), self.color.greenF(), self.color.blueF())
            quadric = gluNewQuadric()
            gluCylinder(quadric, 40, 0.0, 40, 10, 10)
            gluDeleteQuadric(quadric)
            glPopMatrix()
        except Exception as e:
            logging.error(f"Exception during arrow rendering: {e}")


# ================================
# OpenGL Widget
# ================================
class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize dictionary to store plane instances, key is plane ID, value is Plane3DInstance
        self.planes_dict = {}
        # Initialize list to store velocity arrows, list of Arrow3D objects
        self.arrows = []
        # Last mouse position for rotating the view
        self.last_pos = None
        # Initial rotation angles for X and Y axes
        self.rotation_x = 0
        self.rotation_y = 0
        # Initial camera distance and height
        self.camera_distance = 10000.0
        self.camera_height = 10000

        self.camera_target = [0.0, 0.0, 0.0]  # Center point of the scene
        # Model cache dictionary to store loaded models and avoid reloading
        self.plane_model_cache = {}
        # Grid size range, default range is -20000 to 20000
        self.grid_size = 20000
        # Grid line spacing in pixels
        self.grid_spacing = 800

        # 定义敌方战机类型颜色映射
        self.enemy_color_map = {
            0: QColor(60, 150, 255),    # F-16 - DeepSkyBlue
            1: QColor(85, 220, 255),    # Rafale - MidnightBlue
            2: QColor(25, 100, 255),    # Missu - ForestGreen
        }

        # 新增：控制是否显示速度标签
        self.show_speed_labels = True

    def load_plane_model(self, model_path):
        # If model exists in cache, return the cached model
        if model_path in self.plane_model_cache:
            return self.plane_model_cache[model_path]
        # Otherwise, load a new model and store it in cache
        model = Plane3D(model_path, QColor("blue"))
        self.plane_model_cache[model_path] = model
        return model

    def initializeGL(self):
        try:
            glClearColor(1.0, 1.0, 1.0, 1.0)  # White background
            # Enable depth testing for correct rendering order
            glEnable(GL_DEPTH_TEST)
            # Enable color material
            glEnable(GL_COLOR_MATERIAL)
            # Enable lighting
            glEnable(GL_LIGHTING)
            # Enable the first light source
            glEnable(GL_LIGHT0)
            # Enable normalization of normals
            glEnable(GL_NORMALIZE)
            # Set shading model to smooth
            glShadeModel(GL_SMOOTH)

            # Set light position and properties
            glLightfv(GL_LIGHT0, GL_POSITION, [0, 1000, 1000, 1])
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

            # Enable texture mapping
            glEnable(GL_TEXTURE_2D)

            # Log that initialization is complete
            logging.info("OpenGL initialized successfully.")
        except Exception as e:
            # Catch and log any exceptions during initialization
            logging.error(f"OpenGL initialization failed: {e}")

    def resizeGL(self, w, h):
        # Prevent division by zero
        if h == 0:
            h = 1
        # Set the viewport to the window dimensions
        glViewport(0, 0, w, h)
        # Switch to projection matrix mode
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Use perspective projection with a 45-degree field of view
        gluPerspective(45.0, w / h, 1.0, 100000.0)
        # Switch back to modelview matrix mode
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def compute_center_point(self):
        # If there are no planes, return the origin as the center point
        if not self.planes_dict:
            return [0, 0, 0]
        # Calculate the average position of all planes as the scene's center point
        x = sum(plane.position[0] for plane in self.planes_dict.values()) / len(self.planes_dict)
        y = sum(plane.position[1] for plane in self.planes_dict.values()) / len(self.planes_dict)
        z = sum(plane.position[2] for plane in self.planes_dict.values()) / len(self.planes_dict)
        return [x, y, z]

    def paintGL(self):
        try:
            # Clear color and depth buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            # Calculate the center point of the scene
            center = self.compute_center_point()

            # Set the camera position
            gluLookAt(
                self.camera_target[0], self.camera_target[1] + self.camera_height,
                self.camera_target[2] + self.camera_distance,
                self.camera_target[0], self.camera_target[1], self.camera_target[2],
                0, 1, 0
            )

            # Apply scene rotation angles
            glRotatef(self.rotation_x, 1, 0, 0)
            glRotatef(self.rotation_y, 0, 1, 0)

            # Draw coordinate axes with arrows
            self.draw_axes_with_arrows()

            # Draw ground grid
            self.draw_grid()

            # Render all plane instances
            for plane in self.planes_dict.values():
                plane.render()

            # # Render all velocity arrows
            # for arrow in self.arrows:
            #     arrow.render()

            # Finish OpenGL rendering
            glFlush()

            # Use QPainter to draw text labels
            painter = QPainter(self)
            painter.setPen(QColor("#FFFFFF"))  # White text
            painter.setFont(QFont('Arial', 10))
            # 新增：绘制所有飞机的ID标签
            for plane_id, plane_instance in self.planes_dict.items():
                screen_pos = self.world_to_screen(plane_instance.position)
                if screen_pos:
                    x, y = screen_pos
                    # 设置颜色（友军蓝色，敌军红色）
                    label_color = QColor("#00BFFF") if plane_instance.is_enemy else QColor("#FF0000")
                    painter.setPen(label_color)

                    # 在飞机位置上方显示ID（y轴偏移-20）
                    painter.drawText(int(x), int(y) - 20, plane_id)

            # Draw allied planes' command information
            for plane_id, plane_instance in self.planes_dict.items():
                if not plane_instance.is_enemy:  # Only process allied planes
                    plane_pos = plane_instance.position
                    screen_pos = self.world_to_screen([plane_pos[0], plane_pos[1] + 500, plane_pos[2]])  # Raise display position
                    if screen_pos:
                        x, y = screen_pos
                        command = plane_instance.command  # Get command information
                        if command[0] == 0:
                            action_type = "Attack"
                        elif command[0] == 1:
                            action_type = "Evasion"
                        else:
                            action_type = "No Command"

                        if command[1] == -1:
                            target_info = "No Target"
                        else:
                            target_info = f"Group {command[1]}"

                        # Combine command information
                        command_text = f"{action_type} {target_info}"

                        # Draw command information
                        painter.drawText(int(x), int(y), command_text)

            # Iterate through enemy planes and draw group labels and speed
            for plane_id, plane_instance in self.planes_dict.items():
                if plane_instance.is_enemy:  # Only process enemy planes
                    group = plane_instance.group if plane_instance.group is not None else "Unknown"
                    screen_pos = self.world_to_screen(plane_instance.position)
                    if screen_pos:
                        x, y = screen_pos
                        label = f"Group {group}"  # Display group label

                        # **关键修改点：将 'label' 传递给 get_group_index_from_group_label 而不是 'group'**
                        group_index = self.get_group_index_from_group_label(label)
                        color = self.enemy_color_map.get(group_index, QColor("white"))

                        # Set painter color for group label
                        painter.setPen(color)

                        # Draw group label
                        painter.drawText(int(x), int(y), label)

                        # **新增功能：根据是否显示速度标签，绘制速度信息**
                        if self.show_speed_labels:
                            speed = plane_instance.speed
                            speed_text = f"Speed: {speed:.2f}"
                            speed_color = QColor("#FFD700")  # 金色字体
                            painter.setPen(speed_color)
                            painter.drawText(int(x), int(y) + 15, speed_text)  # 速度标签位于组标签下方

            # Draw axis labels
            axis_labels = {"+X": [1000 + 100, 0, 0],
                           "+Y": [0, 1000 + 100, 0],
                           "+Z": [0, 0, 1000 + 100]}
            for label, pos in axis_labels.items():
                screen_pos = self.world_to_screen(pos)
                if screen_pos:
                    x, y = screen_pos
                    painter.drawText(int(x), int(y), label)

            painter.end()

        except Exception as e:
            # Catch and log any exceptions during rendering
            logging.error(f"Exception during rendering: {e}")

    def get_group_index_from_group_label(self, group_label):
        """
        Helper method to extract group index from group label string.
        Assumes group_label is in the format "Group X" where X is the index.
        """
        try:
            # 确保 group_label 是字符串
            if isinstance(group_label, int):
                # 如果 group_label 是整数，直接返回
                return group_label
            return int(group_label.split(" ")[1])
        except (IndexError, ValueError, AttributeError):
            return -1  # Return invalid index if parsing fails

    def world_to_screen(self, position):
        """
        Convert world coordinates to screen coordinates.
        """
        try:
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            win_x, win_y, win_z = gluProject(position[0], position[1], position[2], modelview, projection, viewport)
            return win_x, viewport[3] - win_y  # Flip Y-axis
        except Exception as e:
            logging.error(f"Failed to convert world coordinates to screen coordinates: {e}")
            return None

    def draw_grid(self):
        """
        Draw ground grid located at y=0 plane.
        """
        # Set line width
        glLineWidth(1.0)
        try:
            glColor3f(0.4, 0.4, 0.4)  # Set grid line color to medium gray
            glBegin(GL_LINES)  # Start drawing lines

            # Draw lines parallel to Z-axis
            for x in range(-self.grid_size, self.grid_size + self.grid_spacing, self.grid_spacing):
                glVertex3f(x, 0, -self.grid_size)  # Start point
                glVertex3f(x, 0, self.grid_size)  # End point

            # Draw lines parallel to X-axis
            for z in range(-self.grid_size, self.grid_size + self.grid_spacing, self.grid_spacing):
                glVertex3f(-self.grid_size, 0, z)  # Start point
                glVertex3f(self.grid_size, 0, z)  # End point

            glEnd()  # End drawing
        except Exception as e:
            logging.error(f"Exception during grid drawing: {e}")

    def draw_axes_with_arrows(self, axis_length=800.0, arrow_length=150.0, arrow_radius=40.0):
        """
        Draw 3D coordinate axes with arrows.

        :param axis_length: Length of the axes
        :param arrow_length: Length of the arrowheads
        :param arrow_radius: Radius of the arrowheads
        """
        # Set line width
        glLineWidth(3.0)

        # Start drawing lines
        glBegin(GL_LINES)

        # X-axis - Red
        glColor3f(1.0, 0.0, 0.0)  # Red color
        glVertex3f(0.0, 0.0, 0.0)  # Origin
        glVertex3f(axis_length, 0.0, 0.0)  # X-axis end

        # Y-axis - Green
        glColor3f(0.0, 1.0, 0.0)  # Green color
        glVertex3f(0.0, 0.0, 0.0)  # Origin
        glVertex3f(0.0, axis_length, 0.0)  # Y-axis end

        # Z-axis - Blue
        glColor3f(0.0, 0.0, 1.0)  # Blue color
        glVertex3f(0.0, 0.0, 0.0)  # Origin
        glVertex3f(0.0, 0.0, axis_length)  # Z-axis end

        glEnd()

        # Start drawing arrowheads
        quadric = gluNewQuadric()  # Create a quadric object for drawing cones

        # Draw X-axis arrowhead
        glPushMatrix()  # Save current matrix
        glTranslatef(axis_length, 0.0, 0.0)  # Move to X-axis end
        glRotatef(90, 0, 1, 0)  # Rotate to align cone along X-axis
        glColor3f(1.0, 0.0, 0.0)  # Red color
        gluCylinder(quadric, arrow_radius, 0.0, arrow_length, 32, 32)  # Draw cone
        glPopMatrix()  # Restore matrix

        # Draw Y-axis arrowhead
        glPushMatrix()
        glTranslatef(0.0, axis_length, 0.0)  # Move to Y-axis end
        glRotatef(-90, 1, 0, 0)  # Rotate to align cone along Y-axis
        glColor3f(0.0, 1.0, 0.0)  # Green color
        gluCylinder(quadric, arrow_radius, 0.0, arrow_length, 32, 32)
        glPopMatrix()

        # Draw Z-axis arrowhead
        glPushMatrix()
        glTranslatef(0.0, 0.0, axis_length)  # Move to Z-axis end
        glColor3f(0.0, 0.0, 1.0)  # Blue color
        gluCylinder(quadric, arrow_radius, 0.0, arrow_length, 32, 32)
        glPopMatrix()

        # Delete quadric object to free resources
        gluDeleteQuadric(quadric)

    def get_plane_color_and_arrow_color(self, health_level, base_color):
        """
        Return plane color and arrow color based on health level.
        """
        if health_level > 0:
            return base_color, base_color  # Normal color
        else:
            return QColor("gray"), QColor("gray")  # Gray indicates destroyed

    def update_scene(self, ally_positions, enemy_positions):
        # Clear existing arrows
        self.arrows = []

        # Determine total number of enemy planes
        total_enemies = len(enemy_positions)

        # Preload all models to avoid loading during the loop
        f16_model_path = "models/Meshes/FixedWing.F-16.obj"
        rafale_model_path = "models/Meshes/FixedWing.Rafale.obj"
        missu_model_path = "models/Meshes/FixedWing.MQ-9.obj"

        # Load models and cache them
        f16_model = self.load_plane_model(f16_model_path)
        rafale_model = self.load_plane_model(rafale_model_path)
        missu_model = self.load_plane_model(missu_model_path)

        for index, e in enumerate(enemy_positions):
            plane_id = e.get("id", "unknown")
            pos = [e.get("x", 0), e.get("y", 0), e.get("z", 0)]
            rotation = e.get("rotation", {"pitch": 0, "yaw": 0, "roll": 0})
            vx, vy, vz = e.get("vx", 0), e.get("vy", 0), e.get("vz", 0)
            group = e.get("group", None)  # Only enemies have group info
            health_level = e.get("health_level", 1)  # Get health level
            speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

            # Assign model based on the enemy's index to distribute them into three groups
            if total_enemies > 0:
                group_index = int(index * 3 / total_enemies)
                if group_index >= 3:
                    group_index = 2  # Ensure the index does not exceed 2
            else:
                group_index = 0  # Default to first group if no enemies

            if group_index == 0:
                model = f16_model
            elif group_index == 1:
                model = rafale_model
            else:
                model = missu_model

            # Assign color based on group_index
            plane_color_base = self.enemy_color_map.get(group_index, QColor("blue"))  # Default to blue if not found

            # Adjust color based on health_level
            plane_color, arrow_color = self.get_plane_color_and_arrow_color(health_level, plane_color_base)

            # If the plane doesn't exist, add a new instance
            if plane_id not in self.planes_dict:
                self.planes_dict[plane_id] = Plane3DInstance(
                    model, pos, rotation, plane_color, group=group, is_enemy=True
                )
            else:
                # Update existing plane's position, rotation, and color
                self.planes_dict[plane_id].update_transform(pos, rotation, plane_color, group=group, speed=speed)

            # **关键修改点：设置飞机的速度**
            if plane_id in self.planes_dict:
                self.planes_dict[plane_id].speed = speed

            # Create velocity arrow and add to the list
            if speed > 0:
                direction = [vx / speed, vy / speed, vz / speed]
                arrow = Arrow3D(pos, direction, length=min(max(speed * 5, 40), 5000), color=arrow_color)
                self.arrows.append(arrow)

        # Update allied planes' positions and arrows
        for a in ally_positions:
            plane_id = a.get("id", "unknown")
            pos = [a.get("x", 0), a.get("y", 0), a.get("z", 0)]
            rotation = a.get("rotation", {"pitch": 0, "yaw": 0, "roll": 0})
            vx, vy, vz = a.get("vx", 0), a.get("vy", 0), a.get("vz", 0)
            speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
            command = a.get("command", [-1, -1])  # Get command info
            health_level = a.get("health_level", 1)  # Get health level

            # Set plane and arrow colors based on health level
            plane_color, arrow_color = self.get_plane_color_and_arrow_color(health_level, QColor("red"))

            if plane_id not in self.planes_dict:
                # Assuming allied planes use F-16 model; adjust if different models are needed
                model = self.load_plane_model("models/Meshes/FixedWing.F-16.obj")
                self.planes_dict[plane_id] = Plane3DInstance(
                    model, pos, rotation, plane_color, is_enemy=False
                )
            else:
                self.planes_dict[plane_id].update_transform(pos, rotation, plane_color, command=command, speed=speed)

            # **关键修改点：设置飞机的速度**
            if plane_id in self.planes_dict:
                self.planes_dict[plane_id].speed = speed

            if speed > 0:
                direction = [vx / speed, vy / speed, vz / speed]
                arrow = Arrow3D(pos, direction, length=min(max(speed * 5, 40), 5000), color=arrow_color)
                self.arrows.append(arrow)
        # Refresh the scene
        self.update()

    def mousePressEvent(self, event):
        # Record the mouse press position
        self.last_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.rotation_x = max(-90, min(90, self.rotation_x + dy * 0.2))  # Limit rotation angle
            self.rotation_y += dx * 0.2
            self.update()
        elif event.buttons() & Qt.MouseButton.RightButton:  # Add panning functionality
            self.camera_target[0] -= dx * 0.1
            self.camera_target[2] -= dy * 0.1
            self.update()
        self.last_pos = event.position()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.camera_distance -= delta * 1
        self.camera_distance = max(100.0, min(self.camera_distance, 50000.0))
        self.update()

    def set_show_speed_labels(self, show: bool):
        """
        Set whether to show speed labels.
        """
        self.show_speed_labels = show
        self.update()


# ================================
# Logging Handlers
# ================================

class CustomLogFilter(logging.Filter):
    """
    Custom log filter that only allows logs containing specified keywords to pass.
    """
    def __init__(self, keywords=None):
        super().__init__()
        self.keywords = keywords if keywords else []

    def filter(self, record):
        if not self.keywords:
            return True  # Allow all logs if no keywords are specified
        return any(keyword in record.msg for keyword in self.keywords)


class QTextEditLogger(logging.Handler):
    def __init__(self, parent=None):
        super().__init__()
        self.log_widget = parent

    def emit(self, record):
        msg = self.format(record)
        self.log_widget.append(msg)


# ================================
# Main Window
# ================================

class MainWindow(QMainWindow):
    def __init__(self, client: FrontendClient):
        super().__init__()
        self.client = client

        self.setWindowTitle("Airplane Command & Preference Visualization (3D)")
        self.resize(1920, 1080)

        self.setStyleSheet("""
            /* Global font settings */
            QWidget {
                font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
                font-size: 14px;
                color: #E0E0E0; /* Soft white text */
            }

            /* Background main color */
            QMainWindow {
                background-color: #2C2C2C; /* Darker gray background */
            }

            /* Scrollbar styling */
            QScrollBar:vertical {
                background: #3A3A3A; /* Dark gray background */
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4A90E2, stop:1 #6C6C6C
                ); /* Gradient from blue to gray */
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5FA8F3, stop:1 #8E44AD
                ); /* Hover state gradient blue-purple */
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }

            QScrollBar:horizontal {
                background: #3A3A3A;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4A90E2, stop:1 #6C6C6C
                );
                min-width: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5FA8F3, stop:1 #8E44AD
                );
            }

            /* QLabel styling */
            QLabel {
                color: #E0E0E0; /* Soft white text */
                font-size: 14px;
                font-weight: 500;
            }

            /* QGroupBox styling */
            QGroupBox {
                border: 1px solid #4A4A4A; /* Dark gray border */
                border-radius: 8px;
                margin-top: 10px;
                font-size: 16px;
                font-weight: bold;
                background-color: #333333; /* Slightly darker group background */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px 0 5px;
                color: #4A90E2; /* Blue accent for group titles */
            }

            /* QTableWidget styling */
            QTableWidget {
                /* Removed background-color to prevent overriding cell backgrounds */
                /* background-color: #3A3A3A; */ 
                color: #E0E0E0;
                border: 1px solid #4A4A4A;
                font-size: 12px;
                selection-background-color: #4A90E2; /* Blue selection highlight */
                border-radius: 6px;
            }
            QTableWidget::item {
                /* Ensure item background is transparent so QBrush can be applied */
                background-color: transparent;
            }
            QHeaderView::section {
                background-color: #4A4A4A;
                color: #FFFFFF;
                padding: 6px;
                border: 1px solid #4A4A4A;
                font-size: 12px;
                font-weight: 600;
            }

            /* QPushButton styling */
            QPushButton {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #555555, stop:1 #777777
                ); /* Dark gray gradient background */
                color: #E0E0E0;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                text-align: left;
                padding-left: 10px;
            }
            QPushButton:hover {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4A90E2, stop:1 #6C6C6C
                ); /* Hover state blue-gray gradient */
                color: #FFFFFF; /* White text */
            }

            /* QSlider styling */
            QSlider::groove:horizontal {
                border: 1px solid #6C6C6C; /* Gray border */
                height: 8px;
                background: #E0E0E0; /* Light gray groove */
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4A90E2; /* Blue handle */
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #5FA8F3; /* Lighter blue on hover */
            }

            /* QTextEdit styling */
            QTextEdit {
                background-color: #1E1E1E; /* Dark gray background */
                color: #D4D4D4; /* Soft light gray text */
                border: 1px solid #4A4A4A; /* Dark gray border */
                border-radius: 6px;
                font-size: 12px;
                font-family: 'Courier New', monospace;
            }

            /* Separator styling */
            QFrame {
                background-color: #4A4A4A; /* Separator color */
            }

            /* Additional styles */
        """)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(25)

        # Left Control_CS Panel
        control_layout = QVBoxLayout()
        control_layout.setSpacing(20)  # Increased spacing between modules
        main_layout.addLayout(control_layout, 1)

        # ================================
        # Optimized "Human Preference Settings" Area
        # ================================
        pref_group = QGroupBox("Human Preference Settings")
        pref_layout = QGridLayout()
        pref_group.setLayout(pref_layout)

        # Set padding and spacing for the group box
        pref_layout.setContentsMargins(15, 15, 15, 15)  # Group box padding
        pref_layout.setHorizontalSpacing(25)  # Increased horizontal spacing
        pref_layout.setVerticalSpacing(15)  # Reduced vertical spacing between rows for compactness

        self.preference_widgets = []  # Store sliders and value labels
        action_labels = ["Attack", "Evasion"]  # Row labels
        group_labels = ["Group 0", "Group 1", "Group 2"]  # Column labels

        # Set table headers
        header_label = QLabel("Action \\ Group")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        pref_layout.addWidget(header_label, 0, 0, 1, 1)

        for g_id, g_label in enumerate(group_labels):
            lbl = QLabel(g_label)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 14px; font-weight: bold;")  # More prominent font style
            pref_layout.addWidget(lbl, 0, g_id + 1, 1, 1)

        # Set action labels and slider + value label combinations
        for a_id, a_label in enumerate(action_labels):
            # Add action label
            lbl = QLabel(a_label)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 14px; font-weight: bold;")

            pref_layout.addWidget(lbl, a_id + 1, 0, 1, 1)  # Action label in first column

            for g_id in range(len(group_labels)):
                # Create vertical layout for slider + value label
                slider_layout = QVBoxLayout()

                # Create slider
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, 100)  # Slider range from 0 to 100
                slider.setValue(int(self.client.human_preference[a_id][g_id] * 100))
                slider.setToolTip(f"Set {action_labels[a_id]} preference for {group_labels[g_id]}")

                # Set slider to be horizontally expanding
                slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                slider.setStyleSheet("""
                    QSlider::groove:horizontal {
                        border: 1px solid #6C6C6C;
                        height: 6px;
                        background: #E0E0E0;
                        border-radius: 3px;
                    }
                    QSlider::handle:horizontal {
                        background: #4A90E2;
                        border: 1px solid #4A4A4A;
                        width: 14px;
                        margin: -4px 0;  /* Slide handle protrudes above the groove */
                        border-radius: 7px;
                    }
                    QSlider::handle:horizontal:hover {
                        background: #5FA8F3;
                    }
                """)
                slider.valueChanged.connect(self.real_time_update_preference)

                # Create value label
                value_label = QLabel(f"{self.client.human_preference[a_id][g_id]:.2f}")
                value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                value_label.setStyleSheet("font-size: 12px; color: #4A90E2;")

                # Add slider and value label to vertical layout
                slider_layout.addWidget(slider)  # Slider on top
                slider_layout.addWidget(value_label)  # Value label below

                # Embed vertical layout into a widget and add to grid
                slider_widget = QWidget()
                slider_widget.setLayout(slider_layout)
                pref_layout.addWidget(slider_widget, a_id + 1, g_id + 1, 1, 1)  # Slider + value label combination

                # Store slider and value label
                self.preference_widgets.append((a_id, g_id, slider, value_label))

        # ========== 添加“Reset Preference”、“Reset Camera”和“显示速度”按钮 ==========
        # Create a horizontal layout to hold all three buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)  # Spacing between buttons

        # Create Reset Preference button
        reset_pref_button = QPushButton("Reset Preference")
        reset_pref_button.setToolTip("Click to reset all human preferences to zero")
        reset_pref_button.clicked.connect(self.reset_preferences_to_zero)
        # Load icon using relative path (same as camera icon)
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "camera2.png")
        if os.path.exists(icon_path):
            reset_pref_button.setIcon(QIcon(icon_path))
        else:
            logging.warning(f"Icon file not found: {icon_path}")
        reset_pref_button.setIconSize(QSize(24, 24))
        buttons_layout.addWidget(reset_pref_button)

        # Create Reset Camera button
        reset_camera_button = QPushButton("Reset Camera")
        reset_camera_button.setToolTip("Click to reset the camera view to default position")
        reset_camera_button.clicked.connect(self.reset_camera)
        # Load icon using relative path
        camera_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "camera1.png")
        if os.path.exists(camera_icon_path):
            reset_camera_button.setIcon(QIcon(camera_icon_path))
        else:
            logging.warning(f"Icon file not found: {camera_icon_path}")
        reset_camera_button.setIconSize(QSize(24, 24))
        buttons_layout.addWidget(reset_camera_button)

        # 创建“显示速度”按钮，使用与“重置相机”按钮相同的摄像头图标
        self.toggle_speed_button = QPushButton("Hide Speed")
        self.toggle_speed_button.setToolTip("Click to toggle the display of speed information")
        self.toggle_speed_button.clicked.connect(self.toggle_speed_display)
        # 使用与“重置相机”按钮相同的摄像头图标
        speed_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "speed.png")
        if os.path.exists(speed_icon_path):
            self.toggle_speed_button.setIcon(QIcon(speed_icon_path))
        else:
            logging.warning(f"Icon file not found: {speed_icon_path}")
        self.toggle_speed_button.setIconSize(QSize(24, 24))
        buttons_layout.addWidget(self.toggle_speed_button)

        # Add buttons layout to the preferences grid layout
        pref_layout.addLayout(buttons_layout, len(action_labels) + 1, 0, 1, len(group_labels) + 1)  # Span all columns
        # ===================================

        # Add group box to left control layout
        control_layout.addWidget(pref_group)

        # Add a separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        control_layout.addWidget(separator1)

        # ================================
        # Threat Levels Area
        # ================================
        threat_group = QGroupBox("Threat Levels")
        threat_layout = QVBoxLayout()
        threat_group.setLayout(threat_layout)

        # Create table to display threat levels
        self.threat_table = QTableWidget()
        self.threat_table.setColumnCount(2)
        self.threat_table.setHorizontalHeaderLabels(["Group ID", "Threat Level"])
        self.threat_table.verticalHeader().setVisible(False)
        self.threat_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.threat_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)  # Make table read-only

        threat_layout.addWidget(self.threat_table)

        # Add threat levels group box to left control layout
        control_layout.addWidget(threat_group)

        # Add a separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        control_layout.addWidget(separator2)

        # ================================
        # Status Information Area
        # ================================
        info_group = QGroupBox("Status Information")
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)

        self.connection_label = QLabel("Connection Status: {}".format("Connected" if self.client.connected else "Disconnected"))
        info_layout.addWidget(self.connection_label)

        # Display current human preferences
        pref_table_label = QLabel("Current Human Preferences:")
        pref_table_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        info_layout.addWidget(pref_table_label)

        self.pref_table = QTableWidget(2, 3)
        self.pref_table.setHorizontalHeaderLabels(["Group 0", "Group 1", "Group 2"])
        self.pref_table.setVerticalHeaderLabels(["Attack", "Evasion"])
        self.pref_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pref_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.refresh_preference_table()
        info_layout.addWidget(self.pref_table)

        info_group.setLayout(info_layout)
        control_layout.addWidget(info_group)

        # Add a separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.HLine)
        separator3.setFrameShadow(QFrame.Shadow.Sunken)
        control_layout.addWidget(separator3)

        # Add a spacer to push the logs to the bottom
        control_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Logging Display Area
        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        control_layout.addWidget(log_group)

        # Right Visualization Area (OpenGL)
        plot_layout = QVBoxLayout()
        main_layout.addLayout(plot_layout, 3)

        self.opengl_widget = OpenGLWidget()
        plot_layout.addWidget(self.opengl_widget)

        # Create and start data listener thread (blocking)
        self.listener_thread = DataListenerThread(self.client)
        self.listener_thread.data_received.connect(self.on_data_received)
        self.listener_thread.start()

        # Periodically update connection status (every second)
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_connection_status)
        self.status_timer.start(1000)

        # Set up logging handler to output logs to the log text box
        log_handler = QTextEditLogger(self.log_text)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        log_filter = CustomLogFilter(keywords=["Connection Status", "Camera Reset", "Sent Preferences", "Received update_positions"])
        log_handler.addFilter(log_filter)

        logging.getLogger().addHandler(log_handler)
        logging.getLogger().setLevel(logging.DEBUG)  # Set log level as needed

        # Limit the number of log lines (manual approach)
        self.max_log_lines = 100  # Maximum number of log lines in the log box

        # Start rendering timer (approximately 60FPS)
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.opengl_widget.update)
        self.render_timer.start(16)  # 16ms ≈ 60FPS

    def append_log(self, message):
        """
        Add a message to the log box while limiting the number of log lines.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.log_text.append(f"[{timestamp}] {message}")

        # Check if the number of lines exceeds the maximum
        if self.log_text.document().blockCount() > self.max_log_lines:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)  # Move to the start of the text
            cursor.select(cursor.SelectionType.BlockUnderCursor)  # Select the first line
            cursor.removeSelectedText()  # Remove the selected line
            cursor.deleteChar()  # Remove the paragraph separator

    def set_log_keywords(self, keywords):
        """
        Dynamically set log filtering keywords.
        """
        self.log_filter.keywords = keywords

    def real_time_update_preference(self):
        # Real-time update of preferences
        new_pref = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for a_id, g_id, slider, value_label in self.preference_widgets:
            val = slider.value() / 100.0
            new_pref[a_id][g_id] = val
            value_label.setText(f"{val:.2f}")
        self.client.send_preference(new_pref)
        self.client.human_preference = new_pref
        self.refresh_preference_table()
        self.append_log(f"Sent new preferences: {new_pref}")

    def update_connection_status(self):
        self.connection_label.setText("Connection Status: {}".format("Connected" if self.client.connected else "Disconnected"))

    def on_data_received(self, info):
        if info.get("type") == "update_positions":
            # 检查是否需要重置偏好
            if info.get("reset", False):
                self.reset_human_preferences()
                self.append_log("Received reset signal. Human preferences have been reset.")

            self.client.update_positions(info)
            self.refresh_preference_table()
            self.refresh_threat_table()  # 刷新威胁值表格
            # Update OpenGL visualization
            self.opengl_widget.update_scene(self.client.ally_positions, self.client.enemy_positions)
            # Update log
            num_allies = len(self.client.ally_positions)
            num_enemies = len(self.client.enemy_positions)
            self.append_log(f"Received position update: {num_allies} allies, {num_enemies} enemies.")

    def refresh_preference_table(self):
        pref = self.client.human_preference
        for a_id in range(2):
            for g_id in range(3):
                self.pref_table.setItem(a_id, g_id, QTableWidgetItem(f"{pref[a_id][g_id]:.2f}"))

    def refresh_threat_table(self):
        """
        Refresh the threat levels table based on enemy_groups data with gradient colors.
        """
        enemy_groups = self.client.enemy_groups
        self.threat_table.setRowCount(len(enemy_groups))  # 设置行数
        for row, group in enumerate(enemy_groups):
            group_id = group.get("group_id", "Unknown")
            threat_level = group.get("threat_level", 0.0)

            # 设置Group ID单元格
            item_id = QTableWidgetItem(str(group_id))
            item_id.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.threat_table.setItem(row, 0, item_id)

            # 设置Threat Level单元格
            item_threat = QTableWidgetItem(f"{threat_level:.2f}")
            item_threat.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # 根据威胁值设置背景颜色（绿色到黄色到红色渐变）
            color = self.get_threat_color(threat_level)
            item_threat.setBackground(QBrush(QColor(*color)))
            self.threat_table.setItem(row, 1, item_threat)

    def get_threat_color(self, threat_level):
        """
        根据威胁值返回对应的颜色（绿色到黄色到红色）。
        """
        if threat_level <= 0.3:
            # 绿色到黄色
            ratio = threat_level / 0.3
            red = int(255 * ratio)
            green = 255
            blue = 0
        elif threat_level <= 0.7:
            # 黄色到红色
            ratio = (threat_level - 0.3) / 0.4
            red = 255
            green = int(255 * (1 - ratio))
            blue = 0
        else:
            # 深红色
            red = 255
            green = 0
            blue = 0
        return (red, green, blue)

    def reset_camera(self):
        self.opengl_widget.rotation_x = 0
        self.opengl_widget.rotation_y = 0
        self.opengl_widget.camera_distance = 10000.0
        self.opengl_widget.camera_height = 10000.0
        self.opengl_widget.camera_target = [0.0, 0.0, 0.0]
        self.opengl_widget.update()
        self.append_log("Camera has been reset.")

    def reset_human_preferences(self):
        """
        Reset human preferences to default values when a reset signal is received from the backend.
        """
        default_preferences = [
            [0.0, 0.0, 0.0],  # Attack (0) preferences for Group 0, Group 1, Group 2
            [0.0, 0.0, 0.0]   # Evasion (1) preferences for Group 0, Group 1, Group 2
        ]
        self.client.human_preference = default_preferences
        for a_id, g_id, slider, value_label in self.preference_widgets:
            slider.blockSignals(True)  # Prevent triggering valueChanged signal
            slider.setValue(int(default_preferences[a_id][g_id] * 100))
            slider.blockSignals(False)
            value_label.setText(f"{default_preferences[a_id][g_id]:.2f}")
        self.refresh_preference_table()
        self.append_log(f"Human preferences have been reset to {default_preferences}")

    def reset_preferences_to_zero(self):
        """
        Reset all human preferences to 0 and update the UI.
        """
        default_preferences = [
            [0.0, 0.0, 0.0],  # Attack (0) preferences for Group 0, Group 1, Group 2
            [0.0, 0.0, 0.0]   # Evasion (1) preferences for Group 0, Group 1, Group 2
        ]
        self.client.human_preference = default_preferences
        for a_id, g_id, slider, value_label in self.preference_widgets:
            slider.blockSignals(True)  # Prevent triggering valueChanged signal
            slider.setValue(int(default_preferences[a_id][g_id] * 100))
            slider.blockSignals(False)
            value_label.setText(f"{default_preferences[a_id][g_id]:.2f}")
        self.refresh_preference_table()
        self.client.send_preference(default_preferences)
        self.append_log(f"All human preferences have been reset to {default_preferences}")

    def toggle_speed_display(self):
        """
        Toggle the display of speed information on enemy planes.
        """
        if self.opengl_widget.show_speed_labels:
            self.opengl_widget.set_show_speed_labels(False)
            self.toggle_speed_button.setText("Show Speed")
            self.append_log("Speed display has been hidden.")
        else:
            self.opengl_widget.set_show_speed_labels(True)
            self.toggle_speed_button.setText("Hide Speed")
            self.append_log("Speed display has been shown.")

    def closeEvent(self, event):
        # Stop listener thread and close connection when window closes
        if self.listener_thread.isRunning():
            self.listener_thread.stop()
            self.listener_thread.wait()
        self.client.close()
        event.accept()


# ================================
# Main Entry Point
# ================================

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        client = FrontendClient('127.0.0.1', 9999)
        window = MainWindow(client)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.critical(f"Unhandled exception during program execution: {e}")
