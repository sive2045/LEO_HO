===========================================================================
* SAT 
Elevation Angle: 90 ~ 84 degree
Coverage Area: 55 km (radius)
Rotation Speed: 7.8 km/s

SAT plane: 2
# of SAT per plane: 22 (TBD)
Coordinate:
		First  plane: (0, 0) -> (65, 0) -> ,,, -> (65x, 0)
		Second plane: (25, 65) -> (90, 65) -> ,,, -> (25+65x, 65)
		+ Gaussian Distribution (coord noise)

Episode Time: 0 ~ 155 (s)
===========================================================================
* USER (Ground Basement, fixed)
# of Uesr: 10
Coordinate: (0,0) ~ (100,100); rectangle
===========================================================================
* POMDP
# State: <covering_info, available_channel, visible_time>
# Action: indicator SAT service to GS
# Reward:
	Case 0. -20, Non-service area
	Case 1. -10, HO occur
	Case 2. -5, Overload
	Case 3. visible_time, ACK (MAX 14.10)