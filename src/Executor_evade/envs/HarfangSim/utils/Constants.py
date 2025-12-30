from math import pi

# Print fps function, to check the network client frequency.

t = 0
t0 = 0
t1 = 0

#Aircraft

NormStates = {  # 都不是标准的最大，需要对每一个进行处理
  "speed_max": 400,  # 训练中的飞机的最大速度
  "heading_angle_max": pi,
  "tracking_angle_max": pi / 2,
  "pitch_angle_max": pi / 4,
  "yaw_angle_max": pi,
  "roll_angle_max": pi / 2,
  "AA_max": pi,
  "ATA_max": pi,
  "altitude_max": 10000,
  "health_level": 1,
  "relative_pos_max": 30000,  # 需要根据实际情况进行修改
  "relative_angle_max": 2 * pi,
  "relative_speed_max": 400,  # 需要修改，一般不是800
  "relative_altitude_max": 9000,  # 表示两个飞机的最大绝对值高度距离
  "relative_motion_angle_max": pi,
  "bound_altitude_max": 10000,  # 表示圆柱的最高高度
  "bound_altitude_min": 1000,  # 表示圆柱的最低高度
  "bound_radius": 10000,  # 表示圆柱形区域的最大半径是10km
  "relative_distance_max": 40000  # 表示二者的最大距离

}

#
# NormStates = {
#   "Plane_position"           : 10000,
#   "Plane_Euler_angles"       : 360,
#   "Plane_heading"            : 360,
#   "Plane_pitch_attitude"     : 180,
#   "Plane_roll_attitude"      : 180,
#   "Plane_thrust_level"       : 100,
#   "Plane_horizontal_speed"   : 800,
#   "Plane_vertical_speed"     : 800,
#   "Plane_move_vector"        : 1,
#   "Plane_linear_acceleration": 50,
#   "Missile_position"         : 20000,
#   "Missile_Euler_angles"     : 360,
#   "Missile_move_vector"      : 1,
#   "Missile_heading"          : 360,
#   "Missile_pitch_attitude"   : 180,
#   "Missile_roll_attitude"    : 180,
#   "Missile_horizontal_speed" : 2000,
#   "Missile_vertical_speed"   : 2000
# }
