# -*- coding: utf-8 -*-
import robomaster
from robomaster import robot
from robomaster import gimbal
import time
import numpy as np
import matplotlib.pyplot as plt
import traceback
from collections import deque

class RobotMapper:
    """
    Class for controlling the Robomaster EP robot for exploration and mapping
    using the Frontier-Based Exploration and Conditional Sensor Jig algorithms
    """

    GRID_SIZE_X = 4
    GRID_SIZE_Y = 4
    CELL_RESOLUTION_M = 0.6
    ROBOT_START_X = 1
    ROBOT_START_Y = 1
    ROBOT_START_FACING = 'south'
    MOVE_SPEED = 0.4
    TURN_SPEED = 45
    SLIDE_DISTANCE = 0.1
    SLIDE_SPEED = 0.3
    IR_SENSOR_PORTS = {'left': {'id': 1, 'port': 1}, 'right': {'id': 2, 'port': 1}}
    TOF_WALL_THRESHOLD_CM = 50.0
    LOG_ODDS_OCCUPIED = 0.9
    LOG_ODDS_FREE = -0.7
    LOG_ODDS_MAX = 5.0
    LOG_ODDS_MIN = -5.0
    
    # DIRECTION_VECTORS ให้คงเดิมสำหรับการแสดงผลในแมพ
    DIRECTION_VECTORS = {'north': (0, -1), 'south': (0, 1), 'east': (1, 0), 'west': (-1, 0)}
    
    # แก้ไข TURNS table ให้ตรงกับการเดินจริงของหุ่นยนต์ (สลับ left-right)
    TURNS = {'north': {'left': 'east', 'right': 'west', 'u_turn': 'south'},
             'east':  {'left': 'south', 'right': 'north', 'u_turn': 'west'},  
             'south': {'left': 'west', 'right': 'east', 'u_turn': 'north'},
             'west':  {'left': 'north', 'right': 'south', 'u_turn': 'east'}}

    #-------------------------------------------------------------------------------------------------------------


    def __init__(self):
        self.ep_robot = None
        self.ep_chassis = None
        self.ep_sensor = None
        self.sensor_adaptor = None
        self.position = [self.ROBOT_START_X, self.ROBOT_START_Y]
        self.facing = self.ROBOT_START_FACING
        self.occupancy_grid = np.zeros((self.GRID_SIZE_Y, self.GRID_SIZE_X))
        self.visited_cells = set()
        self.unvisited_frontier = []
        self.path_history = []
        self.wall_positions = set()
        self.wall_edges = set()
        self.current_robot_position = [0, 0]  # สำหรับ PID control
        self.start_position = tuple(self.position)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

    def _get_stable_ir_reading(self, port_info, samples=5):
        """Reads IR values multiple times and returns the most frequent value"""
        readings = []
        for _ in range(samples):
            readings.append(self.sensor_adaptor.get_io(**port_info))
            time.sleep(0.02) # Small delay between readings

        # Return the most frequent value (0 or 1)
        return max(set(readings), key=readings.count)

    def _conditional_slide_scan(self):
        """
        Perform a conditional slide scan by sliding sideways to confirm sensor values
        Returns sensor data collected from multiple points (safest)
        """
        print("  - Performing conditional slide scan...")
        
        # 1. Scan at the center position first
        center_readings = self._read_sensors_only()
        all_readings = {'center': center_readings}
        
        # 2. Check if sliding is necessary - only when values are uncertain
        need_slide_left = (center_readings.get('left', -1) == -1 or 
                          center_readings.get('left') == 0)
        need_slide_right = (center_readings.get('right', -1) == -1 or 
                           center_readings.get('right') == 0)
        
        # 3. Slide left (if necessary) - reduced time
        if need_slide_left:
            print(f"    - Quick slide left {self.SLIDE_DISTANCE}m...")
            self.ep_chassis.move(x=0, y=-self.SLIDE_DISTANCE, z=0, xy_speed=self.SLIDE_SPEED).wait_for_completed()
            time.sleep(0.2)  # Reduced delay
            left_readings = self._read_sensors_only()
            all_readings['left_slide'] = left_readings
            
            # Return to center
            self.ep_chassis.move(x=0, y=self.SLIDE_DISTANCE, z=0, xy_speed=self.SLIDE_SPEED).wait_for_completed()
            time.sleep(0.2)
        
        # 4. Slide right (if necessary) - reduced time
        if need_slide_right:
            print(f"    - Quick slide right {self.SLIDE_DISTANCE}m...")
            self.ep_chassis.move(x=0, y=self.SLIDE_DISTANCE, z=0, xy_speed=self.SLIDE_SPEED).wait_for_completed()
            time.sleep(0.2)
            right_readings = self._read_sensors_only()
            all_readings['right_slide'] = right_readings
            
            # Return to center
            self.ep_chassis.move(x=0, y=-self.SLIDE_DISTANCE, z=0, xy_speed=self.SLIDE_SPEED).wait_for_completed()
            time.sleep(0.2)
        
        # 5. Collect and summarize the safest results
        final_readings = self._consolidate_readings(all_readings)
        
        print(f"    - Consolidated readings: {final_readings}")
        return final_readings

    def _consolidate_readings(self, all_readings):
        """
        Consolidate data from multiple scans and select the safest value
        (if blocked was detected even once, consider it blocked)
        """
        consolidated = {}
        sensors = ['front', 'left', 'right']
        
        for sensor in sensors:
            readings = []
            for scan_name, reading_dict in all_readings.items():
                if sensor in reading_dict and reading_dict[sensor] != -1:
                    readings.append(reading_dict[sensor])
            
            if readings:
                # If values were read - choose the safest one (1=occupied is safer than 0=free)
                consolidated[sensor] = max(readings)
            else:
                # If no readings were possible
                consolidated[sensor] = -1
        
        return consolidated

    def _read_sensors_only(self):
        """Read sensor values only, no print or delay"""
        states = {}
        
        # Read IR sensors
        for direction, port_info in self.IR_SENSOR_PORTS.items():
            try:
                stable_value = self._get_stable_ir_reading(port_info, samples=3)
                states[direction] = 1 if stable_value == 0 else 0
            except:
                states[direction] = -1
        
        # Read ToF sensor  
        try:
            tof_readings_mm = []
            def tof_callback(tof_info):
                if len(tof_readings_mm) < 3:
                    tof_readings_mm.append(tof_info[0])
            
            self.ep_sensor.sub_distance(freq=20, callback=tof_callback)
            start_time = time.time()
            while len(tof_readings_mm) < 3 and time.time() - start_time < 0.8:
                time.sleep(0.05)
            self.ep_sensor.unsub_distance()
            
            if tof_readings_mm:
                distance_cm = (sum(tof_readings_mm) / len(tof_readings_mm)) / 10.0
                states['front'] = 1 if distance_cm < self.TOF_WALL_THRESHOLD_CM else 0
            else:
                states['front'] = -1
        except:
            states['front'] = -1
        
        return states

    def _position_handler(self, position_info):
        """Callback function สำหรับรับตำแหน่งของหุ่นยนต์"""
        self.current_robot_position = position_info[:2]

    def _move_forward_one_grid_with_pid(self):
        """เดินหน้าหนึ่งช่องด้วย PID controller เพื่อความแม่นยำ"""
        print(f"  - Action: Moving forward one grid ({self.CELL_RESOLUTION_M}m) with PID control.")
        
        # รอให้ได้ตำแหน่งเริ่มต้น
        while self.current_robot_position == [0, 0]:
            time.sleep(0.1)
        
        start_x, start_y = self.current_robot_position
        target_distance = self.CELL_RESOLUTION_M
        
        # PID parameters
        kp, ki, kd = 2.5, 0.1, 0.3
        integral, prev_error = 0, 0
        
        start_time = time.time()
        max_time = 8.0  # เพิ่มเวลาสูงสุด
        
        print(f"    - PID: Starting from ({start_x:.3f}, {start_y:.3f}), target distance: {target_distance:.3f}m")
        
        while time.time() - start_time < max_time:
            # คำนวณระยะทางที่เดินได้
            current_x, current_y = self.current_robot_position
            distance_moved = np.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
            
            # คำนวณ error
            error = target_distance - distance_moved
            
            print(f"    - PID: Distance moved: {distance_moved:.3f}m, Error: {error:.3f}m")
            
            # ถ้าถึงเป้าหมายแล้ว
            if abs(error) < 0.02:  # tolerance 2cm
                print("    - PID: Target reached!")
                break
            
            # PID calculation
            integral += error * 0.1
            derivative = (error - prev_error) / 0.1
            
            # คำนวณความเร็ว
            speed = kp * error + ki * integral + kd * derivative
            speed = np.clip(speed, 0, self.MOVE_SPEED)
            
            # ส่งคำสั่งให้หุ่นเดิน
            self.ep_chassis.drive_speed(x=speed)
            
            prev_error = error
            time.sleep(0.1)
        
        # หยุดหุ่นยนต์
        self.ep_chassis.drive_speed(x=0)
        print("  - PID: Movement completed.")
        time.sleep(1)

    def _update_frontier_list(self, sensor_readings):
        """
        Update the frontier list (unvisited paths) 
        For example: if at (3,3) and can go 2 ways: (2,3) and (3,4)
        We will go forward (2,3) and keep (3,4) in the frontier
        """
        current_pos = tuple(self.position)
        
        # Find all possible directions
        possible_directions = []
        sensor_to_direction = {'front': 'forward', 'left': 'left', 'right': 'right'}
        
        for sensor_key, direction in sensor_to_direction.items():
            if sensor_readings.get(sensor_key) == 0:  # Path is free
                target_pos = self._get_target_cell_in_direction(current_pos, self.facing, direction)
                if (self._is_valid_position_1based(target_pos) and 
                    tuple(target_pos) not in self.visited_cells):
                    possible_directions.append((direction, target_pos))
        
        # If more than 1 way, keep the unselected ways in the frontier
        if len(possible_directions) > 1:
            print(f"  - Found {len(possible_directions)} possible directions: {possible_directions}")
            
            # Prefer going forward first (if possible)
            front_option = None
            other_options = []
            
            for direction, target_pos in possible_directions:
                if direction == 'forward':
                    front_option = target_pos
                else:
                    other_options.append(target_pos)
            
            # Add other options to the frontier
            for pos in other_options:
                if pos not in self.unvisited_frontier:
                    self.unvisited_frontier.append(pos)
                    print(f"    - Added ({pos[0]},{pos[1]}) to frontier list")
            
            # If no forward option, add the first one to the frontier
            if not front_option and other_options:
                self.unvisited_frontier.extend(other_options[1:])  # Add remaining ways
                return other_options[0]  # Return the first one as the target
                
            return front_option
        elif len(possible_directions) == 1:
            return possible_directions[0][1]
        
        return None

    def _get_target_cell_in_direction(self, pos, facing, relative_direction):
        if relative_direction == 'forward':
            target_facing = facing
        else:
            target_facing = self.TURNS[facing][relative_direction]
        dx, dy = self.DIRECTION_VECTORS[target_facing]
        return (pos[0] + dx, pos[1] + dy)

    def _get_required_facing_to_target(self, current_pos, target_pos):
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        for direction, (vec_x, vec_y) in self.DIRECTION_VECTORS.items():
            if vec_x == dx and vec_y == dy:
                return direction
        return None

    def _find_shortest_path_to_target(self, target):
        """Use BFS to find the shortest path to the target"""
        if tuple(self.position) == target:
            return []
        
        queue = deque([(tuple(self.position), [])])
        visited = {tuple(self.position)}
        
        while queue:
            current_pos, path = queue.popleft()
            
            for direction, (dx, dy) in self.DIRECTION_VECTORS.items():
                neighbor = (current_pos[0] + dx, current_pos[1] + dy)
                
                if (neighbor not in visited and 
                    self._is_valid_position_1based(neighbor) and
                    neighbor not in self.wall_positions):
                    
                    new_path = path + [neighbor]
                    
                    if neighbor == target:
                        return new_path
                    
                    queue.append((neighbor, new_path))
                    visited.add(neighbor)
        
        return None

    @staticmethod
    def _log_odds_to_probability(grid):
        return 1 - (1 / (1 + np.exp(grid)))

    def _move_forward_one_grid(self):
        print(f"  - Action: Moving forward one grid ({self.CELL_RESOLUTION_M}m).")
        self.ep_chassis.move(x=self.CELL_RESOLUTION_M, y=0, z=0, xy_speed=self.MOVE_SPEED).wait_for_completed()
        time.sleep(1)

    def _turn_to_facing(self, target_facing):
        if self.facing == target_facing:
            return
            
        # แก้ไขมุมการหมุนให้สอดคล้องกับ DIRECTION_VECTORS
        # หุ่นยนต์หมุนทวนเข็มนาฬิกาเป็นบวก
        facing_angles = {'north': 0, 'east': 90, 'south': 180, 'west': -90}
        current_angle = facing_angles[self.facing]
        target_angle = facing_angles[target_facing]
        
        angle_to_turn = target_angle - current_angle
        if angle_to_turn > 180: angle_to_turn -= 360
        if angle_to_turn < -180: angle_to_turn += 360

        print(f"  - Action: Turning from {self.facing} to {target_facing} ({angle_to_turn} degrees).")
        self.ep_chassis.move(x=0, y=0, z=angle_to_turn, z_speed=self.TURN_SPEED).wait_for_completed()
        
        print("  - Action: Recenter gimbal to follow chassis.")
        self.ep_robot.gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()

        self.facing = target_facing
        time.sleep(0.5)

    def _update_occupancy_grid(self, sensor_readings):
        print("  - Updating Occupancy Grid and Wall positions...")
        sensor_to_direction_map = {'front': 'forward', 'left': 'left', 'right': 'right'}
        
        for sensor_key, state in sensor_readings.items():
            if state not in [0, 1]: continue
            relative_dir = sensor_to_direction_map.get(sensor_key)
            if not relative_dir: continue
            
            tx, ty = self._get_target_cell_in_direction(self.position, self.facing, relative_dir)
            if self._is_valid_position_1based((tx, ty)):
                # แปลงพิกัด 1-based เป็น 0-based สำหรับ occupancy_grid
                grid_x, grid_y = tx - 1, ty - 1
                update = self.LOG_ODDS_OCCUPIED if state == 1 else self.LOG_ODDS_FREE
                current_log_odd = self.occupancy_grid[grid_y, grid_x]
                new_log_odd = np.clip(current_log_odd + update, self.LOG_ODDS_MIN, self.LOG_ODDS_MAX)
                self.occupancy_grid[grid_y, grid_x] = new_log_odd
                
                # Add wall to the list if an obstacle is detected
                if state == 1:
                    self.wall_positions.add((tx, ty))
                    # เพิ่มขอบกำแพงเฉพาะตามทิศทางที่เจอ
                    self._add_wall_edge_by_direction(self.position, (tx, ty), relative_dir)
                    print(f"    - Wall detected at ({tx}, {ty})")

    def _add_wall_edge_by_direction(self, robot_pos, wall_pos, direction):
        """เพิ่มขอบกำแพงให้สอดคล้องกับ TURNS ที่แก้ไขแล้ว"""
        rx, ry = robot_pos
        wx, wy = wall_pos
        
        # แปลงเป็นพิกัดการแสดงผล (ลบ 1)
        robot_draw_x, robot_draw_y = rx - 1, ry - 1
        wall_draw_x, wall_draw_y = wx - 1, wy - 1
        
        edge = None
        
        # วาดขอบกำแพงตามทิศทางที่หุ่นยนต์เจอ - ปรับให้สอดคล้องกับ TURNS ที่แก้ไขแล้ว
        if direction == 'forward':
            # กำแพงข้างหน้า
            if self.facing == 'north':
                edge = ((robot_draw_x, robot_draw_y), (robot_draw_x + 1, robot_draw_y))
            elif self.facing == 'south':
                edge = ((robot_draw_x, robot_draw_y + 1), (robot_draw_x + 1, robot_draw_y + 1))
            elif self.facing == 'east':
                edge = ((robot_draw_x + 1, robot_draw_y), (robot_draw_x + 1, robot_draw_y + 1))
            elif self.facing == 'west':
                edge = ((robot_draw_x, robot_draw_y), (robot_draw_x, robot_draw_y + 1))
                
        elif direction == 'left':
            # กำแพงข้างซ้าย - แก้ไขให้ตรงกับ TURNS ใหม่
            if self.facing == 'north':  # หันเหนือ ซ้ายคือทิศตะวันออก (แก้ไขแล้ว)
                edge = ((robot_draw_x + 1, robot_draw_y), (robot_draw_x + 1, robot_draw_y + 1))
            elif self.facing == 'south':  # หันใต้ ซ้ายคือทิศตะวันตก (แก้ไขแล้ว)
                edge = ((robot_draw_x, robot_draw_y), (robot_draw_x, robot_draw_y + 1))
            elif self.facing == 'east':  # หันตะวันออก ซ้ายคือทิศใต้ (แก้ไขแล้ว)
                edge = ((robot_draw_x, robot_draw_y + 1), (robot_draw_x + 1, robot_draw_y + 1))
            elif self.facing == 'west':  # หันตะวันตก ซ้ายคือทิศเหนือ (แก้ไขแล้ว)
                edge = ((robot_draw_x, robot_draw_y), (robot_draw_x + 1, robot_draw_y))
                
        elif direction == 'right':
            # กำแพงข้างขวา - แก้ไขให้ตรงกับ TURNS ใหม่
            if self.facing == 'north':  # หันเหนือ ขวาคือทิศตะวันตก (แก้ไขแล้ว)
                edge = ((robot_draw_x, robot_draw_y), (robot_draw_x, robot_draw_y + 1))
            elif self.facing == 'south':  # หันใต้ ขวาคือทิศตะวันออก (แก้ไขแล้ว)
                edge = ((robot_draw_x + 1, robot_draw_y), (robot_draw_x + 1, robot_draw_y + 1))
            elif self.facing == 'east':  # หันตะวันออก ขวาคือทิศเหนือ (แก้ไขแล้ว)
                edge = ((robot_draw_x, robot_draw_y), (robot_draw_x + 1, robot_draw_y))
            elif self.facing == 'west':  # หันตะวันตก ขวาคือทิศใต้ (แก้ไขแล้ว)
                edge = ((robot_draw_x, robot_draw_y + 1), (robot_draw_x + 1, robot_draw_y + 1))
        
        if edge:
            self.wall_edges.add(edge)
            print(f"    - Added wall edge: {edge} (direction: {direction}, facing: {self.facing})")
            
            # Debug: แสดงการแปลงทิศทาง
            if direction in ['left', 'right']:
                expected_wall_direction = self.TURNS[self.facing][direction]
                print(f"    - Debug: Robot facing {self.facing} + sensor {direction} = wall at {expected_wall_direction}")

    def _visualize_map(self):
        plt.clf()
        
        # Create a light gray background (unknown areas)
        prob_grid = self._log_odds_to_probability(self.occupancy_grid)
        display_grid = np.full((self.GRID_SIZE_Y, self.GRID_SIZE_X, 3), [0.8, 0.8, 0.8])
        
        # Show free spaces (white)
        display_grid[prob_grid < 0.45] = [1, 1, 1]
        
        # Show background
        self.ax = plt.gca()
        self.ax.imshow(display_grid, origin='upper', extent=[0, self.GRID_SIZE_X, self.GRID_SIZE_Y, 0])
        
        # Draw wall edges (thick solid black lines)
        wall_count = 0
        for edge in self.wall_edges:
            (x1, y1), (x2, y2) = edge
            # วาดเส้นกำแพงหนาและทึบ
            self.ax.plot([x1, x2], [y1, y2], 'k-', linewidth=8, solid_capstyle='round')
            wall_count += 1
        
        print(f"  - Plotted {wall_count} wall edges")
        
        # Draw path (purple line connecting start to current)
        if len(self.path_history) > 0:
            all_path_points = [self.start_position] + self.path_history + [tuple(self.position)]
            path_x = [p[0] - 0.5 for p in all_path_points]
            path_y = [p[1] - 0.5 for p in all_path_points]
            
            self.ax.plot(path_x, path_y, 'purple', linewidth=3, alpha=0.7, 
                        linestyle='-', marker='o', markersize=4, markerfacecolor='purple')
        
        # Draw start position (red dot)
        start_x, start_y = self.start_position
        self.ax.plot(start_x - 0.5, start_y - 0.5, 'ro', markersize=12, markeredgecolor='darkred', markeredgewidth=2)
        
        # Draw current position (green dot)
        rx, ry = self.position
        self.ax.plot(rx - 0.5, ry - 0.5, 'go', markersize=15, markeredgecolor='darkgreen', markeredgewidth=2)
        
        # แสดงทิศทางที่หุ่นยนต์กำลังหัน (ลูกศร) - ใช้ DIRECTION_VECTORS
        arrow_length = 0.3
        dx, dy = self.DIRECTION_VECTORS[self.facing]
        dx *= arrow_length
        dy *= arrow_length
            
        self.ax.arrow(rx - 0.5, ry - 0.5, dx, dy, 
                     head_width=0.1, head_length=0.1, fc='green', ec='darkgreen', linewidth=2)
        
        # Set axis and grid
        self.ax.set_xlim(0, self.GRID_SIZE_X)
        self.ax.set_ylim(0, self.GRID_SIZE_Y)
        self.ax.set_xticks(np.arange(0, self.GRID_SIZE_X + 1, 1))
        self.ax.set_yticks(np.arange(0, self.GRID_SIZE_Y + 1, 1))
        
        # เพิ่ม labels บนแกนให้เป็น 1-based coordinates
        self.ax.set_xticklabels(['', '1', '2', '3', '4'])
        self.ax.set_yticklabels(['', '1', '2', '3', '4'])
        
        self.ax.grid(True, color='gray', alpha=0.3)
        self.ax.set_title("🗺️ Robot Exploration Map - Corrected Left/Right\n🟢=Current 🔴=Start 🟣=Path ⬛=Wall Lines ➡️=Direction", fontsize=14)
        
        # Show additional information with direction details
        frontier_info = f"Frontier: {len(self.unvisited_frontier)} cells"
        wall_edges_info = f"Wall Edges: {len(self.wall_edges)} detected"
        facing_info = f"Facing: {self.facing.upper()}"
        
        # แสดงการแปลงทิศทาง left/right สำหรับ debug
        left_turn = self.TURNS[self.facing]['left']
        right_turn = self.TURNS[self.facing]['right']
        turn_info = f"L→{left_turn}, R→{right_turn}"
        
        info_text = (f"Position: ({self.position[0]},{self.position[1]})\n"
                    f"{facing_info}\n"
                    f"Turns: {turn_info}\n"
                    f"Visited: {len(self.visited_cells)} cells\n"
                    f"Walls: {len(self.wall_positions)} detected\n"
                    f"{wall_edges_info}\n"
                    f"{frontier_info}")
        
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        # Add TURNS table guide
        turns_text = ("TURNS Table (Fixed):\n"
                     "North: L→East, R→West\n"
                     "East: L→South, R→North\n"
                     "South: L→West, R→East\n"
                     "West: L→North, R→South")
        self.ax.text(0.98, 0.98, turns_text, transform=self.ax.transAxes,
                    verticalalignment='top', horizontalalignment='right', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        # Add color legend
        legend_text = "🟢 Robot + Direction Arrow\n🔴 Start Point (1,1)\n🟣 Travel Path\n⬛ Wall Lines (L/R Fixed)\n📏 PID Movement"
        self.ax.text(0.98, 0.02, legend_text, transform=self.ax.transAxes,
                    verticalalignment='bottom', horizontalalignment='right', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
        
        plt.draw()
        plt.pause(0.1)

    def run(self):
        """Main function to start all operations"""
        try:
            self.ep_robot = robot.Robot()
            self.ep_robot.initialize(conn_type="ap")
            print("✅ Robot connected.")
            self.ep_chassis = self.ep_robot.chassis
            self.ep_sensor = self.ep_robot.sensor
            self.sensor_adaptor = self.ep_robot.sensor_adaptor
            
            # เริ่ม position tracking สำหรับ PID
            self.ep_chassis.sub_position(freq=10, callback=self._position_handler)
            print("✅ Position tracking started for PID control.")
            
            # Center Gimbal to face forward only once at the start
            self.ep_robot.gimbal.recenter(yaw_speed=500).wait_for_completed()
            
            # Start exploration
            start_pos = tuple(self.position)
            self.visited_cells.add(start_pos)
            print(f"\n🚀 Starting Smart Exploration with PID at position ({self.position[0]},{self.position[1]})")
            print(f"Initial facing: {self.facing}")
            
            step_count = 0
            max_steps = (self.GRID_SIZE_X * self.GRID_SIZE_Y) * 2

            while step_count < max_steps:
                print(f"\n--------- Step {step_count + 1} ---------")
                print(f"Current: ({self.position[0]},{self.position[1]}), Facing: {self.facing}")
                print(f"Frontier list: {self.unvisited_frontier}")
                
                # 1. Scan the environment with Conditional Slide Scan
                sensor_readings = self._conditional_slide_scan()
                self._update_occupancy_grid(sensor_readings)
                
                # 2. Update frontier and find the next target
                next_target = self._update_frontier_list(sensor_readings)
                
                if next_target and tuple(next_target) not in self.visited_cells:
                    # There is a new available path - move to it
                    print(f"  - Moving to new target: ({next_target[0]},{next_target[1]})")
                    required_facing = self._get_required_facing_to_target(self.position, next_target)
                    
                    if required_facing:
                        self._turn_to_facing(required_facing)
                        old_position = tuple(self.position)
                        self._move_forward_one_grid_with_pid()  # ใช้ PID แทน
                        self.position = list(next_target)
                        self.path_history.append(old_position)
                        self.visited_cells.add(tuple(next_target))
                        
                elif self.unvisited_frontier:
                    # No nearby paths, but there are unvisited frontiers
                    target_frontier = self.unvisited_frontier.pop(0)
                    print(f"  - Going to frontier: ({target_frontier[0]},{target_frontier[1]})")
                    
                    path_to_target = self._find_shortest_path_to_target(target_frontier)
                    if path_to_target and len(path_to_target) > 0:
                        next_step = path_to_target[0]
                        required_facing = self._get_required_facing_to_target(self.position, next_step)
                        
                        if required_facing:
                            self._turn_to_facing(required_facing)
                            old_position = tuple(self.position)
                            self._move_forward_one_grid_with_pid()  # ใช้ PID แทน
                            self.path_history.append(old_position)
                            self.position = list(next_step)
                            self.visited_cells.add(tuple(next_step))
                    else:
                        print(f"  - Cannot reach frontier ({target_frontier[0]},{target_frontier[1]})")
                else:
                    print("  - No more frontiers found. Exploration complete!")
                    break
                
                # 3. Update the map
                self._visualize_map()
                step_count += 1
            
            print("\n✅ Smart Exploration with PID finished.")
            
        except KeyboardInterrupt:
            print("\n🛑 Program interrupted by user.")
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            traceback.print_exc()
        finally:
            if self.ep_robot:
                self.ep_chassis.drive_speed(x=0)
                self.ep_chassis.unsub_position()  # หยุด position tracking
                filename = "final_occupancy_grid.csv"
                prob_grid = self._log_odds_to_probability(self.occupancy_grid)
                np.savetxt(filename, prob_grid, delimiter=",", fmt="%.3f")
                print(f"💾 Final map saved to '{filename}'")
                self.ep_robot.close()
                print("\n🔌 Robot connection closed.")
            print("Displaying final map. Close the window to exit.")
            plt.ioff()
            self._visualize_map()
            plt.show()

    def _is_valid_position_1based(self, pos):
        """Check if the position is within the 1-based coordinate map boundaries"""
        x, y = pos
        return 1 <= x <= self.GRID_SIZE_X and 1 <= y <= self.GRID_SIZE_Y

    def _update_frontier_list(self, sensor_readings):
        """Update frontier list using 1-based coordinates"""
        current_pos = tuple(self.position)
        
        # Find all possible directions
        possible_directions = []
        sensor_to_direction = {'front': 'forward', 'left': 'left', 'right': 'right'}
        
        for sensor_key, direction in sensor_to_direction.items():
            if sensor_readings.get(sensor_key) == 0:  # Path is free
                target_pos = self._get_target_cell_in_direction(current_pos, self.facing, direction)
                if (self._is_valid_position_1based(target_pos) and 
                    tuple(target_pos) not in self.visited_cells):
                    possible_directions.append((direction, target_pos))
        
        # If more than 1 way, keep the unselected ways in the frontier
        if len(possible_directions) > 1:
            print(f"  - Found {len(possible_directions)} possible directions: {possible_directions}")
            
            # Prefer going forward first (if possible)
            front_option = None
            other_options = []
            
            for direction, target_pos in possible_directions:
                if direction == 'forward':
                    front_option = target_pos
                else:
                    other_options.append(target_pos)
            
            # Add other options to the frontier
            for pos in other_options:
                if pos not in self.unvisited_frontier:
                    self.unvisited_frontier.append(pos)
                    print(f"    - Added ({pos[0]},{pos[1]}) to frontier list")
            
            # If no forward option, add the first one to the frontier
            if not front_option and other_options:
                self.unvisited_frontier.extend(other_options[1:])  # Add remaining ways
                return other_options[0]  # Return the first one as the target
                
            return front_option
        elif len(possible_directions) == 1:
            return possible_directions[0][1]
        
        return None

if __name__ == '__main__':
    mapper = RobotMapper()
    mapper.run()