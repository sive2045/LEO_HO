"""
# Low Earth Orbit Satellites Handover

|----------------------|-----------------------------------------------------------------------|
| Author               | Chungneung Lee                                                        |
|----------------------|-----------------------------------------------------------------------|
| Actions              | Select SAT                                                            |
| Agents               | `agents= ['ground_station_0', 'ground_station_1', ...']`              |
| Agents               | # of ground stations                                                  |
| Action Shape         | (# of SAT * # of Plane,)                                              |
| Action Values        | [0, 1]                                                                |
| Observation Shape    | (3, # of SAT * # of Plane)                                            |
| Observation Values   | (0, 1) or (visible time (s))                                          |
|----------------------|-----------------------------------------------------------------------|

### Version History

* v0: Initial versions release (1.0.0)
* v1: Add interfence SAT, considering SINR (1.1.0)
      -> Trajectory: y = x 

TODO
1. 클래스 파라메터 추가 (채널 개수, 로드 등등)
2. 위성 위치 에러 추가 고려하기 -> 완료
3. 구체적인 채널 파라미터도 고려해야 함 -> SINR 계산 추가해야함.
--> SINR, 선택에 따라 값이 바뀌는지 확인이 필요함. ***************************
"""
import matplotlib.pyplot as plt
from copy import copy

import numpy as np
from gymnasium import spaces

from pettingzoo.utils import agent_selector
from pettingzoo import AECEnv

class LEOSATEnv(AECEnv):
    def __init__(self, render_mode=None, debugging=False, interference_mode=True) -> None:        
        #|------Agent args--------------------------------------------------------------------------------------------------------------------------|
        self.GS_area_max_x = 100    # km
        self.GS_area_max_y = 100    # km
        self.GS_area_max_z = 1_000  # km

        self.GS_size = 10
        self.GS = np.zeros((self.GS_size, 3)) # coordinate (x, y, z) of GS
        self.GS_speed = 0.167 # km/s -> 60 km/h
        self.anttena_gain = 30 # [dBi]
        self.shadow_fading = 0.5 # [dB]
        self.GS_Tx_power = 23e-3 # 23 dBm

        self.timestep = None
        self.terminal_time = 155 # s
        #|------SAT(serviced) args------------------------------------------------------------------------------------------------------------------|
        self.SAT_len = 22
        self.SAT_plane = 2 # of plane
        self.SAT_coverage_radius = 55 # km
        self.SAT_speed = 6.87 # km/s, caution!! direction
        self.theta = np.linspace(0, 2 * np.pi, 150)
        self.SAT_point = np.zeros((self.SAT_len * self.SAT_plane, 3)) # coordinate (x, y, z) of SAT center point
        self.SAT_coverage = np.zeros((self.SAT_len * self.SAT_plane, 3, 150)) # coordinate (x, y, z) of SAT coverage
        self.SAT_height = 500 # km, SAT height
        self.SAT_point[:,2] = self.SAT_height # km, SAT height 
        self.SAT_coverage[:,2,:] = self.SAT_height # km, SAT height
        self.SAT_Load = np.full(self.SAT_len*self.SAT_plane, 5) # the available channels of SAT
        self.SAT_BW = 10 # MHz BW budget of SAT
        self.freq = 14 # GHz

        self.SAT_Tx_power = 11 # dBw ---> 수정 필요
        self.GNSS_noise = 1 # GNSS measurement noise, Gaussian white noise
        
        self.SINR_weight = 2 # SINR reward weight
        self.load_weight = 1 # Remaining load reward weight

        self.service_indicator = np.zeros((self.GS_size, self.SAT_len*self.SAT_plane)) # indicator: users are served by SAT (one-hot vector)
        #|------SAT(interference) args--------------------------------------------------------------------------------------------------------------|
        self.interference_mode = interference_mode
        self.ifc_SAT_len = 22
        self.ifc_SAT_speed = 7.07 # km/s
        self.ifc_SAT_point = np.zeros((self.ifc_SAT_len, 3)) # coordinate (x, y, z) of interference SAT center point
        self.ifc_SAT_coverage = np.zeros((self.ifc_SAT_len, 3, 150)) # coordinate (x, y, z) of interference SAT coverage point
        self.ifc_SAT_coverage_radius = 70 # km
        self.ifc_SAT_height = 700 # km
        self.ifc_SAT_point[:,2] = self.ifc_SAT_height
        self.ifc_SAT_coverage[:,2,:] = self.ifc_SAT_height
        self.ifc_SAT_BW = 10 # MHz BW
        self.ifc_Tx_power = 8 # dBw ----> 수정 해야함
        self.ifc_freq = 15 # GHz ---> 주파수 살짝만 다름 고려사항 생각
        self.ifc_service_indicator = np.zeros((self.GS_size, self.ifc_SAT_len)) # indicator: users are in interference SAT
        #|------Agent args--------------------------------------------------------------------------------------------------------------------------|
        self.agents = [f"groud_station_{i}" for i in range(self.GS_size)]
        self.possible_agents = self.agents[:]

        self._none = self.SAT_len * self.SAT_plane
        self.action_spaces = {i: spaces.Discrete(self.SAT_len * self.SAT_plane) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(4, self.SAT_len * self.SAT_plane), dtype=np.int8
                    ),
                }
            )
            for i in self.agents
        }

        self.render_mode = render_mode        

        self.debugging = debugging

    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _GS_random_walk(self, GS, speed):
        """
        Update GS poistion by ramdom walk.
            speed [km/s]
            time [s]

        return GS position
        """
        _GS = np.copy(GS)
        for i in range(len(_GS)):
            val = np.random.randint(1,4)
            if val == 1:
                _GS[i,0] += speed
            elif val == 2:
                _GS[i,0] -= speed
            elif val == 3:
                _GS[i,1] += speed
            else:
                _GS[i,1] -= speed
        
        return _GS

    def _SAT_coordinate(self, SAT, SAT_len, time, speed):
        """
        return real-time SAT center point
        """
        _SAT = np.copy(SAT)
        for i in range(SAT_len):
            _SAT[i,0] = 65*i -speed * time + np.random.normal(self.GNSS_noise)
            _SAT[i,1] = 10 + np.random.normal(self.GNSS_noise)
            _SAT[i,2] = self.SAT_height + np.random.normal(self.GNSS_noise)

            _SAT[i + SAT_len,0] = -25 + 65*i -speed * time + np.random.normal(self.GNSS_noise)
            _SAT[i + SAT_len,1] =  10 + 65 + np.random.normal(self.GNSS_noise)
            _SAT[i + SAT_len,2] = self.SAT_height + np.random.normal(self.GNSS_noise)
        
        if self.interference_mode:
            for i in range(self.ifc_SAT_len):
                self.ifc_SAT_point[i,0] = 80 - 100/np.sqrt(2) * i + self.ifc_SAT_speed/np.sqrt(2) * time + np.random.normal(self.GNSS_noise)
                self.ifc_SAT_point[i,1] = 80 - 100/np.sqrt(2) * i + self.ifc_SAT_speed/np.sqrt(2) * time + np.random.normal(self.GNSS_noise)
                self.ifc_SAT_point[i,2] = self.ifc_SAT_height + np.random.normal(self.GNSS_noise)

        return _SAT

    def _SAT_coverage_position(self, SAT_point, SAT_coverage, SAT_len, time, speed, radius, theta):
        """
        return real-time SAT coverage position
        for render
        """
        _SAT_coverage = np.copy(SAT_coverage)
        for i in range(SAT_len):
            _SAT_coverage[i,0,:] = SAT_point[i,0] + radius * np.cos(theta)
            _SAT_coverage[i,1,:] =  10                + radius * np.sin(theta)

            _SAT_coverage[i + SAT_len,0,:] = SAT_point[i + SAT_len, 0] + radius * np.cos(theta)
            _SAT_coverage[i + SAT_len,1,:] =  10 +   65               + radius * np.sin(theta)

        if self.interference_mode:
            for i in range(self.ifc_SAT_len):
                self.ifc_SAT_coverage[i,0,:] = self.ifc_SAT_point[i,0] + self.ifc_SAT_coverage_radius * np.cos(theta)
                self.ifc_SAT_coverage[i,1,:] = self.ifc_SAT_point[i,1] + self.ifc_SAT_coverage_radius * np.sin(theta)

        return _SAT_coverage

    def _is_in_coverage(self, SAT, GS, coverage_radius):
        """
        return coverage indicator (one-hot vector)
        """
        dist = np.zeros((len(GS), len(SAT)))
        coverage_indicator = np.zeros((len(GS), len(SAT)))

        for i in range(len(GS)):
            for j in range(len(SAT)):
                dist[i][j] = np.linalg.norm(GS[i,0:2] - SAT[j,0:2]) # 2-dim 
        
        coverage_index = np.where(dist <= coverage_radius)
        
        coverage_indicator[coverage_index[:][0], coverage_index[:][1]] = 1

        if self.interference_mode:
            _dist = np.zeros((self.GS_size, self.ifc_SAT_len))
            for i in range(self.GS_size):
                for j in range(self.ifc_SAT_len):
                    _dist[i][j] = np.linalg.norm(GS[i,0:2] - self.ifc_SAT_point[j,0:2]) # 2-dim
            
            _ifc_coverage_idx = np.where(_dist <= self.ifc_SAT_coverage_radius)
            self.ifc_service_indicator[_ifc_coverage_idx[:][0], _ifc_coverage_idx[:][1]] = 1

        return coverage_indicator        

    def _get_visible_time(self, SAT_point, SAT_speed, coverage_radius, GS):
        """
        return visible time btw SAT and GS
        """
        _num = np.max((coverage_radius ** 2 - (GS[1]-SAT_point[1]) ** 2, 0))
        visible_time = (np.sqrt(_num) - GS[0] + SAT_point[0]) / SAT_speed
        visible_time = np.max((visible_time, 0))
        visible_time = 0 if visible_time >= 14 else visible_time
        return visible_time        


    def _cal_signal_power(self, SAT, GS, freq):
        """
        기본적으로 거리가 길어서 path loss를 HO에서 SINR를 고려하기 좀 애매함.

        return uplink signal power

        shadow faiding loss -> avg 1
        """
        GS_signal_power = np.zeros((len(GS), len(SAT)))
        for i in range(len(GS)):
            for j in range(len(SAT)):
                dist = np.linalg.norm(GS[i,:] - SAT[j,:])
                #print(f"{self.timestep}시간, {i}유저와 {j}위성 간 거리 : {dist}") 
                if GS[i,0] > SAT[j,0]:
                    delta_f = (1 + self.SAT_speed / 3e5) * freq
                else:
                    delta_f = (1 - self.SAT_speed / 3e5) * freq
                #FSPL = ( np.pi * 4 * dist * delta_f) ** -2 # Power, free space path loss;
                FSPL = 20 * np.log10(dist) + 20 * np.log10(delta_f) + 92.45 # [dB], free space path loss
                GS_signal_power[i, j] =  self.SAT_Tx_power - FSPL - self.shadow_fading + 30
                #GS_signal_power[i, j] =  self.SAT_Tx_power * FSPL * self.shadow_fading * 30
        if self.debugging: print(f"{self.timestep}-times Agents' signal power: {GS_signal_power}")
        return GS_signal_power

    def _cal_SINR(self, GS_index, signal_power, SAT_service_idx, noise_temperature = 550):
        """
        Input parameter:
            noise_temperature: 550 [K]
        return downlink SINR
        """
        SINR = 0 # [dB]

        # noise
        noise_power = 10 * np.log10(noise_temperature / 290 + 1) # [dB]
        #noise = 10 **(0.1*noise_power)

        # communication constellation interfernce
        comm_ifc = np.sum(signal_power) - signal_power[SAT_service_idx] # dB
        # interference constellation
        # SAT_ifc = np.zeros(self.ifc_SAT_len) # dB 디버깅용, 연산 소모때문에 학습땐 지양.
        SAT_ifc = 0
        for i in range(self.ifc_SAT_len):
            dist = np.linalg.norm(self.GS[GS_index,:] - self.ifc_SAT_point[i,:])
            #print(f"{self.timestep}시간, {GS_index}유저와 {i}간섭위성 간 거리 : {dist}")
            if self.GS[GS_index,0] > self.ifc_SAT_point[i,0]:
                delta_f = (1 + self.ifc_SAT_speed / 3e5) * self.ifc_freq
            else:
                delta_f = (1 - self.ifc_SAT_speed / 3e5) * self.ifc_freq
            #FSPL = ( np.pi * 4 * dist * delta_f) ** -2 # Power, free space path loss;
            FSPL = 20 * np.log10(dist) + 20 * np.log10(delta_f) + 92.45 # [dB], free space path loss
            SAT_ifc += self.ifc_Tx_power - (FSPL + self.shadow_fading) + 30 # 디버깅 시 SAT_ifc[i]로 변환! # 1000 -> Antenna gain
            #SAT_ifc += self.ifc_Tx_power * FSPL * self.shadow_fading * 30

        # SINR calculate
        SINR = signal_power[GS_index] - comm_ifc - SAT_ifc - noise_power # dB
        #SINR = 10*np.log(signal_power[SAT_service_idx] / (comm_ifc + SAT_ifc + noise))
        if self.debugging: print(f"{self.timestep}-times {GS_index}-Agent, comm ifc: {comm_ifc}\nSAT ifc: {SAT_ifc}")
        return SINR

    def reset(self, seed=None, return_info=False, options=None):
        self.timestep = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.states = {agent: self._none for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # set GS position
        for i in range(self.GS_size):
            self.GS[i][0] = np.random.randint(0, self.GS_area_max_x + 1)
            self.GS[i][1] = np.random.randint(0, self.GS_area_max_y + 1)

        # set SAT position
        self.SAT_point = self._SAT_coordinate(self.SAT_point, self.SAT_len, self.timestep, self.SAT_speed)
        # coverage indicator
        self.coverage_indicator = self._is_in_coverage(self.SAT_point, self.GS, self.SAT_coverage_radius)
        # visible time
        self.visible_time = np.zeros((self.GS_size,self.SAT_len*2))
        for i in range(self.GS_size):
            for j in range(self.SAT_len*2):
                self.visible_time[i][j] = self._get_visible_time(self.SAT_point[j], self.SAT_speed, self.SAT_coverage_radius, self.GS[i])
        # Update SINR info, 모든 정보?? 확인 필요! 
        if self.interference_mode:
            SINRs = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane)) # <----------------- SINRs 클래스 init 여부 고민!
            SINRs_avg = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane)) # SINRs - avg 값
            signal_power = self._cal_signal_power(self.SAT_point, self.GS, self.freq)
            for i in range(self.GS_size):
                for j in range(self.SAT_len * self.SAT_plane):
                    SINRs[i][j] = self._cal_SINR(i, signal_power[i,:], j)
                SINRs_avg[i] = SINRs[i] - np.average(SINRs[i])
        # observations
        self.observations = {}
        for i in range(self.GS_size):
            observation = (
                self.coverage_indicator[i],
                self.SAT_Load,
                self.visible_time[i],
                SINRs_avg[i]
            )
            self.observations[self.agents[i]] = observation

        #return self.observations
    
    def observe(self, agent):
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        # Execute actions and Get Rewards
        # Action must select a covering SAT
        agent = self.agent_selection
        if self.debugging:  print(f"timestep:{self.timestep}, agent: {agent}")        
        self.states[self.agent_selection] = action
        if self.debugging: print(f"timestep:{self.timestep}, state : {self.states[agent]}")
        
        if self._agent_selector.is_last():
            # Update service indicator
            _service_indicator = np.copy(self.service_indicator)
            self.service_indicator = np.zeros((self.GS_size, self.SAT_len*self.SAT_plane))
            for i in range(self.GS_size):
                self.service_indicator[i][self.states[self.agents[i]]] = 1
    
            # Update SAT position
            self.SAT_point = self._SAT_coordinate(self.SAT_point, self.SAT_len, self.timestep, self.SAT_speed)
            # Update GS position
            self.GS = self._GS_random_walk(self.GS, self.GS_speed)
            if self.debugging:  print(f"timestep:{self.timestep}, agent poistion: {self.GS}")            
            # Update coverage indicator
            self.coverage_indicator = self._is_in_coverage(self.SAT_point, self.GS, self.SAT_coverage_radius)
            # Get visible time        
            for i in range(self.GS_size):
                for j in range(self.SAT_len*2):
                    self.visible_time[i][j] = self._get_visible_time(self.SAT_point[j], self.SAT_speed, self.SAT_coverage_radius, self.GS[i])
            # Update SINR info
            if self.interference_mode:
                SINRs = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane)) # <----------------- SINRs 클래스 init 여부 고민!
                SINRs_avg = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane)) # SINRs - avg 값
                signal_power = self._cal_signal_power(self.SAT_point, self.GS, self.freq)
                for i in range(self.GS_size):
                    for j in range(self.SAT_len * self.SAT_plane):
                        SINRs[i][j] = self._cal_SINR(i, signal_power[i,:], j)
                    SINRs_avg[i] = SINRs[i] - np.average(SINRs[i])

            for i in range(self.GS_size):
                observation = (
                    self.coverage_indicator[i],
                    self.SAT_Load,
                    self.visible_time[i],
                    SINRs_avg[i]
                )
                self.observations[f"groud_station_{i}"] = observation
            
            # rewards
            for i in range(self.GS_size):
                reward = 0

                # non-coverage area
                if self.coverage_indicator[i][self.states[self.agents[i]]] == 0:
                    reward = -50
                # HO occur
                elif _service_indicator[i][self.states[self.agents[i]]] == 0:
                    reward = -30
                else:
                # Overload
                    _actions = np.array(list(self.states.values()))
                    if np.count_nonzero(_actions == _actions[i]) > self.SAT_Load[_actions[i]]:
                        reward = -25
                    else:
                        if self.interference_mode:                         
                            SINR = float(SINRs_avg[i, np.where(self.service_indicator[i] == 1)])
                            reward = self.visible_time[i][_actions[i]] + self.load_weight * (self.SAT_Load[_actions[i]] - np.count_nonzero(_actions == _actions[i])) + self.SINR_weight * 10 ** (0.1*(SINR))
                            if self.debugging: print(f"ACK Status with SINR mode, {i}-th GS, Selected SAT: {_actions[i]}, Remaining load: {(self.SAT_Load[_actions[i]] - np.count_nonzero(_actions == _actions[i]))}, SINR reward: {self.SINR_weight * 10 ** (0.1*(SINR))}")
                        else:
                            reward = self.visible_time[i][_actions[i]] + self.load_weight * (self.SAT_Load[_actions[i]] - np.count_nonzero(_actions == _actions[i]))
                            if self.debugging: print(f"ACK Status, {i}-th GS, Selected SAT: {_actions[i]}, Remaining load: {(self.SAT_Load[_actions[i]] - np.count_nonzero(_actions == _actions[i]))}")
                self.rewards[self.agents[i]] = reward
            if self.debugging: print(f"rewards:{self.rewards},\n visible_time: {self.visible_time}]\nSINR: {SINRs}\nSINR_avg: {SINRs_avg}") # 디버깅시 SINR도 보이게 설정.

            if self.render_mode == "human":
                self.render()

            # Check termination conditions
            if self.timestep == self.terminal_time:
                self.terminations = {agent: True for agent in self.agents}
            
            # Get obersvations
            self.timestep += 1
        else:
            self._clear_rewards()
        
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        #return self.observations, self.rewards, self.terminations, self.truncations, self.infos
        
    def render(self):
        """
        Caution time step !!
        -> execute this func before step func
        """
        figure, axes = plt.subplots(1)
        
        SAT_area = self._SAT_coverage_position(self.SAT_point, self.SAT_coverage, self.SAT_len, self.timestep, self.SAT_speed, self.SAT_coverage_radius, self.theta)        

        # Plot SATs' coverage area
        for i in range(self.SAT_len * self.SAT_plane):
            axes.plot(SAT_area[i,0,:], SAT_area[i,1,:], color='#2B3467')
        # Plot SATs' point
        axes.plot(self.SAT_point[:,0], self.SAT_point[:,1], 'o')
        
        # Plot ground stations' area
        axes.plot([0,100,100,0,0], [0,0,100,100,0])
        # Plot ground stations
        axes.plot(self.GS[:,0], self.GS[:,1], '*')
        # Plot Selected line        
        for i in range(self.GS_size): # train 단에서 버그발견 수정 해야함
            axes.plot(
                (self.GS[i,0], self.SAT_point[self.states[self.agents[i]],0]),
                (self.GS[i,1], self.SAT_point[self.states[self.agents[i]],1]),
                "--k", linewidth=1
                )
        
        if self.interference_mode:
            # Plot Interference SATs' point
            axes.plot(self.ifc_SAT_point[:,0], self.ifc_SAT_point[:,1], 's')
            # Plot Interference SAT's coverage
            for i in range(self.ifc_SAT_len):
                axes.plot(self.ifc_SAT_coverage[i,0,:], self.ifc_SAT_coverage[i,1,:], color='#EB455F', linestyle=':')

        axes.set_aspect(1)
        axes.axis([-50, 200, -50, 150])

        plt.show()