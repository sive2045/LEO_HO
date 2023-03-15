# Mult-Agent Deep Reinforcement Learning Based LEO Satellites Handover Solution 🌍
## Abstract 📚
```
Author: Chungneung Lee
Goal: 5G NR 저궤도 위성망에서 핸드오버에 의해 발생하는 다양한 통신 문제를 
	  MADRL을 통해 해결. (TBD)
```
---
## Simulation Parameters 🔧
### Satellite 📡
```
Elevation Angle: 90 ~ 84º
Coverage Area: 55 km (radius)
Rotation Speed: 7.8 km/s
Number of SAT plane: 2
Number of SAT per plane: 22
Coordinate:
		First  plane: (0, 0) -> (65, 0) -> ,,, -> (65x, 0)
		Second plane: (25, 65) -> (90, 65) -> ,,, -> (25+65x, 65)
		+ Gaussian Distribution (coord noise)

Episode Time: 0 ~ 155 (s)
```
### User Equipment (Ground Basement) 📱
```
USER (Ground Basement, fixed) --> It will add mobility
Number of Uesr: 10
Users' Coordinate: (0,0) ~ (100,100); randomly distributed
```
### Partial Observation Markov Decision Process (POMDP) 🔭
```
State: <covering_info, available_channel, visible_time>
Action: indicator SAT service to GS
Reward:
	Case 0. -30, Non-service area
	Case 1. -10, HO occur
	Case 2. -5, Overload
	Case 3. visible_time, ACK (MAX 14.10)
```