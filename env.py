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
      Add Banchmark Scheme (1.1.1)

TODO
1. 결과 그래프 
2. 벤치 마킹 스킴 추가 구현, 세부 결정
"""
import matplotlib.pyplot as plt
from copy import copy

from itertools import groupby
import random
import numpy as np
from gymnasium import spaces
import scipy.special as sc

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
        self.shadow_fading = 0.5 # [dB]
        self.GS_Tx_power = 23e-3 # 23 dBm
        self.threshold = -2 # [dB]

        self.timestep = None
        self.terminal_time = 155 # s
        
        self.rate_threshold = 100_000_000 # 100 Mbps
        self.data_rate = np.zeros((self.GS_size))
        self.rate_data_log = np.zeros((self.GS_size, self.terminal_time + 1))

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
        self.SAT_Load_MAX = np.full(self.SAT_len*self.SAT_plane, 50) # the maximum available channels of SAT
        self.SAT_Load = np.zeros((self.SAT_len * self.SAT_plane)) # the available channels of SAT
        self.load_info = np.zeros((self.GS_size, self.SAT_len*self.SAT_plane)) # load info
        
        self.SAT_BW = 200_000_000 # Hz BW budget of SAT
        self.freq = 20 # GHz
        self.SAT_Tx_power = 40 # W
        self.GNSS_noise = 1 # 수정 예정
        self.anttena_gain = 1_000
        
        self.visible_time_weight = 1
        self.rate_weight = 10**(-8)

        self.SINR_weight = 1 # SINR reward weight
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
        self.ifc_freq = 16 # GHz ---> 고려사항 생각
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
        #|------Debugging args-----------------------------------------------------------------------------------------------------------------------|
        self.render_mode = render_mode        
        self.debugging = debugging
        
        # Result vars
        # Ack, Blocked, HOF
        # --> non-coverage는 어떻게 뺄지 고민해야함.
        self.agent_status_log = np.zeros((self.GS_size, self.terminal_time+1)) # 1: non-serviced, 2: HOF-QoS, 3: HO, 4: ACK
        self.SINR_log = np.zeros((self.GS_size, self.terminal_time+1))
        self.load_log = np.zeros((self.GS_size, self.terminal_time+1))

    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def init_benchmark_scheme(self):
        """
        Benchmark Scheme

        1. Maximum Visible Time (MVT)
        2. Maximum Available Channels (MAC)
        3. SINR-based 
        """
        # MVT
        self.MVT_status_log = np.zeros((self.GS_size, self.terminal_time+1)) # 1: non-serviced, 2: HOF-QoS, 3: HO, 4: ACK
        self.MVT_service_index = np.zeros((self.GS_size)) # SAT index
        self.MVT_service_index = np.random.randint(0, self.SAT_len*self.SAT_plane, (self.SAT_len*self.SAT_plane)) # init index
        self.MVT_data_rate_log = np.zeros((self.GS_size, self.terminal_time+1))
        self.MVT_SAT_Load_MAX = np.full(self.SAT_len*self.SAT_plane, 50) # the maximum available channels of SAT
        self.MVT_SAT_Load = np.zeros((self.SAT_len * self.SAT_plane)) # the available channels of SAT
        
        # random
        self.random_status_log = np.zeros((self.GS_size, self.terminal_time+1)) # 1: non-serviced, 2: HOF-QoS, 3: HO, 4: ACK
        self.random_service_index = np.zeros((self.GS_size)) # SAT index
        self.random_service_index = np.random.randint(0, self.SAT_len*self.SAT_plane, self.SAT_len*self.SAT_plane) # init index
        self.random_data_rate_log = np.zeros((self.GS_size, self.terminal_time+1))
        self.random_SAT_Load_MAX = np.full(self.SAT_len*self.SAT_plane, 50) # the maximum available channels of SAT
        self.random_SAT_Load = np.zeros((self.SAT_len * self.SAT_plane)) # the available channels of SAT

        # channel-gain-based
        self.channel_based_status_log = np.zeros((self.GS_size, self.terminal_time+1)) # 1: non-serviced, 2: HOF-QoS, 3: HO, 4: ACK
        self.channel_based_service_index = np.zeros((self.GS_size)) # SAT index
        self.channel_based_service_index = np.random.randint(0, self.SAT_len*self.SAT_plane, self.SAT_len*self.SAT_plane) # init index
        self.channel_based_data_rate_log = np.zeros((self.GS_size, self.terminal_time+1))
        self.channel_gaind_based_SAT_Load_MAX = np.full(self.SAT_len*self.SAT_plane, 50) # the maximum available channels of SAT
        self.channel_gaind_based_SAT_Load = np.zeros((self.SAT_len * self.SAT_plane)) # the available channels of SAT

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

    def _select_random_index(self, recv_indices) -> list:
        '''
        if overloaded state occurs,
        return SAT allocate indices list and overload indices list
        --> [selected_list, remaining_list]
        '''
        print(f"받은 인덱스 :{recv_indices}")
        selected = random.sample(recv_indices, self.SAT_Load_MAX[recv_indices[0][0]])
        selected_list = list(selected)
        remaining_list = [item for item in recv_indices if item not in selected]
        return [selected_list, remaining_list]

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
        (for render)
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

    def _cal_shadowed_rice_fading_gain(self):
        self.channel_gain = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane))
        for i in range(self.GS_size):
            for j in range(self.SAT_len * self.SAT_plane):
                if self.coverage_indicator[i][j] == 1:
                    elevation_angle = 180/np.pi * np.arctan(self.SAT_height / (np.sqrt( (self.GS[i][0]-self.SAT_point[j][0])**2 + (self.GS[i][1]-self.SAT_point[j][1])**2 )))
                    # paramters
                    b = (-4.7943 * 10**(-8) * elevation_angle**(3) + 5.5784 * 10**(-6) * elevation_angle**(2) 
                         -2.1344 * 10**(-4) * elevation_angle + 3.2710 * 10**(-2))
                    # LoS componets
                    omega = (
                        1.4428 * 10**(-5) * elevation_angle**(3) -2.3798 * 10**(-3) * elevation_angle**(2)
                        +1.2702 * 10**(-1) * elevation_angle -1.4864
                    )
                    # Nakagami-m fading
                    m = (
                        6.3739 * 10**(-5) * elevation_angle**(3) +5.8533 * 10**(-4) * elevation_angle**(2)
                        -1.5973 * 10**(-1) * elevation_angle + 3.5156
                    )
                    x = np.linspace(0,4,1000)
                    PDF = 1/(2*b) * (((2*b*m)/(2*b*m+omega))**(m)) * np.exp(-x/(2*b)) * sc.hyp1f1(m,1,omega*x/(2*b*(2*b*m+omega)))
                    idx = np.random.randint(0,1000)
                    self.channel_gain[i, j] = PDF[idx]
        if self.debugging: print(f"{self.timestep}-times Agents' channel gain: {self.channel_gain}")
        return self.channel_gain

    def _cal_data_rate(self, actions, SAT_load):
        data_rate = np.zeros((self.GS_size))
        rate_data_log = np.zeros((self.GS_size, self.terminal_time + 1))
        for i in range(self.GS_size):
            if self.coverage_indicator[i][actions[i]]==0: pass
            else:
                interfernce = 1
                for j in range(self.SAT_len * self.SAT_plane):
                    if actions[i] == j: pass
                    elif self.coverage_indicator[i][j] == 1 and self.SAT_Load[j] > 0:
                        interfernce += self.SAT_Tx_power*self.anttena_gain*self.channel_gain[i][j]
                
                FSPL = (299792458/(4*np.pi*self.freq*1000000*np.linalg.norm(self.GS[i] - self.SAT_point[actions[i]]))) ** 2
                data_rate[i] = self.SAT_BW/SAT_load[actions[i]] * np.log2(1 + (self.SAT_Tx_power*self.anttena_gain*FSPL*self.channel_gain[i][actions[i]] / (interfernce*FSPL + 10**(-12))))
        if self.debugging: print(f"{self.timestep}-times Agents' data rate: {self.data_rate}")
        return data_rate

    def _update_load_SAT(self) -> None:
        '''
        time slot 한번당 실행
        -> 초기화 후 사용
        '''
        self.SAT_Load = np.zeros((self.SAT_len * self.SAT_plane))
        for i in range(self.GS_size):
            self.SAT_Load[self.states[self.agents[i]]] += 1
    
    def _update_benchmark_load_info(self, actions):
        '''
        벤치마킹 load 정보 업데이트
        '''
        SAT_Load = np.zeros((self.SAT_len * self.SAT_plane))
        for i in range(self.GS_size):
            SAT_Load[actions[i]] += 1
        return SAT_Load
    
    # load 정보 update 이상함?
    def _get_load_SAT(self) -> None:
        self.load_info = np.zeros((self.GS_size, self.SAT_len*self.SAT_plane))
        for gs_idx in range(self.GS_size):
            _load_info = np.zeros((self.SAT_len*self.SAT_plane))
            for SAT_idx in range(self.SAT_len*self.SAT_plane):
                if self.coverage_indicator[gs_idx][SAT_idx]:
                    _load_info[SAT_idx] = np.max((0, self.SAT_Load_MAX[SAT_idx] - self.SAT_Load[SAT_idx]))
                else: _load_info[SAT_idx] = 0
            self.load_info[gs_idx] = _load_info

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
        # Update channel gain
        self.channel_gain = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane))

        # observations
        self.observations = {}
        for i in range(self.GS_size):
            observation = (
                self.coverage_indicator[i],
                self.load_info[i],
                self.visible_time[i],
                self.channel_gain[i]
            )
            self.observations[self.agents[i]] = observation
        
        # logs
        self.agent_status_log = np.zeros((self.GS_size, self.terminal_time+1)) # 1: non-serviced, 2: HOF-QoS, 3: HO, 4: ACK 
        self.SINR_log = np.zeros((self.GS_size, self.terminal_time+1))
        self.load_log = np.zeros((self.GS_size, self.terminal_time+1))

        # Benchmark Scheme
        if self.debugging:
            self.init_benchmark_scheme()

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
            # visible time
            self.visible_time = np.zeros((self.GS_size,self.SAT_len*2))
            for i in range(self.GS_size):
                for j in range(self.SAT_len*2):
                    self.visible_time[i][j] = self._get_visible_time(self.SAT_point[j], self.SAT_speed, self.SAT_coverage_radius, self.GS[i])
            # Update load info
            self._update_load_SAT()
            # Get load info
            self._get_load_SAT()
            # Update channel gain
            self._cal_shadowed_rice_fading_gain()
            
            _actions = np.array(list(self.states.values()))
            #print(f"{self.timestep}-step actions: {_actions}")
            # Update Data rate
            self.data_rate = self._cal_data_rate(_actions, self.SAT_Load)
            self.rate_data_log[:,self.timestep] =  self.data_rate

            for i in range(self.GS_size):
                observation = (
                    self.coverage_indicator[i],
                    self.load_info[i],
                    self.visible_time[i],
                    self.channel_gain[i]
                )
                self.observations[f"groud_station_{i}"] = observation
                
                # non-coverage area
                if self.coverage_indicator[i][self.states[self.agents[i]]] == 0:
                    reward = -100
                    self.agent_status_log[i][self.timestep] = 1
                    _actions[i] = -1 # overload 카운팅에서 제외 설정
                    self.service_indicator[i] = np.zeros(self.SAT_len*self.SAT_plane) # 다음 time slot에 무조건 HO가 일어나도록 설정; 대기 상태
                    self.rewards[self.agents[i]] = reward

            # update benchmark data rate
            if self.debugging:
                MVT_data_rate = np.zeros((self.GS_size))
                random_data_rate = np.zeros((self.GS_size))
                channle_gain_based_data_rate = np.zeros((self.GS_size))

                # get load info
                MVT_load_info = self._update_benchmark_load_info(self.MVT_service_index)
                random_load_info = self._update_benchmark_load_info(self.random_service_index)
                channle_based_load_info = self._update_benchmark_load_info(self.channel_based_service_index)

                MVT_data_rate = self._cal_data_rate(self.MVT_service_index, MVT_load_info)
                self.MVT_data_rate_log[:, self.timestep] = MVT_data_rate
                random_data_rate = self._cal_data_rate(self.random_service_index, random_load_info)
                self.random_data_rate_log[:, self.timestep] = random_data_rate
                channle_gain_based_data_rate = self._cal_data_rate(self.channel_based_service_index, channle_based_load_info)
                self.channel_based_data_rate_log[:, self.timestep] = channle_gain_based_data_rate

            # rewards
            for i in range(self.GS_size):
                # Benchmark
                if self.debugging:                     
                    print(f"{self.timestep}-time step, {i}-th agent info: MVT: {self.MVT_service_index[i]}")
                    print(f"{self.timestep}-time step, {i}-th agent info: RANDOM: {self.random_service_index[i]}")
                    print(f"{self.timestep}-time step, {i}-th agent info: channel-gain-based: {self.channel_based_service_index[i]}")
                    # MVT
                    if self.coverage_indicator[i][self.MVT_service_index[i]] == 0 or MVT_data_rate[i] < self.rate_threshold:
                        idx = np.where(self.visible_time[i] == np.max(self.visible_time[i]))[0][0]
                        interference = 0
                        for j in range(self.SAT_len * self.SAT_plane):
                            if idx == j: continue
                            elif self.coverage_indicator[i][j] == 1 and MVT_load_info[j] > 0:
                                    interference += self.SAT_Tx_power*self.anttena_gain*self.channel_gain[i][j]
                        FSPL = (299792458/(4*np.pi*self.freq*1000000*np.linalg.norm(self.GS[i] - self.SAT_point[idx]))) ** 2
                        data_rate = self.SAT_BW/MVT_load_info[idx] * np.log2(1 + (self.SAT_Tx_power*self.anttena_gain*FSPL*self.channel_gain[i][idx] / (interference*FSPL + 10**(-12))))
                        # HOF: QoS
                        if data_rate < self.rate_threshold:
                            self.MVT_status_log[i][self.timestep] = 2
                        # HO
                        else:
                            self.MVT_service_index[i] = idx
                            self.MVT_status_log[i][self.timestep] = 3
                    else:
                        self.MVT_status_log[i][self.timestep] = 4
                    
                    # RANDOM
                    if self.coverage_indicator[i][self.random_service_index[i]] == 0 or random_data_rate[i] < self.rate_threshold:
                        idx = np.random.randint(0, self.SAT_len*self.SAT_plane)
                        interference = 0
                        for j in range(self.SAT_len * self.SAT_plane):
                            if idx == j: continue
                            elif self.coverage_indicator[i][j] == 1 and random_load_info[j] > 0:
                                    interference += self.SAT_Tx_power*self.anttena_gain*self.channel_gain[i][j]
                        FSPL = (299792458/(4*np.pi*self.freq*1000000*np.linalg.norm(self.GS[i] - self.SAT_point[idx]))) ** 2
                        data_rate = self.SAT_BW/random_load_info[idx] * np.log2(1 + (self.SAT_Tx_power*self.anttena_gain*FSPL*self.channel_gain[i][idx] / (interference*FSPL + 10**(-12))))
                        
                        if self.coverage_indicator[i][idx] == 0:
                            self.random_status_log[i][self.timestep] = 1
                        elif data_rate < self.rate_threshold:
                            self.random_status_log[i][self.timestep] = 2
                        # HO
                        else:
                            self.random_service_index[i] = idx
                            self.random_status_log[i][self.timestep] = 3
                    else:
                        self.random_status_log[i][self.timestep] = 4

                    # channel gain-based
                    if self.coverage_indicator[i][self.channel_based_service_index[i]] == 0 or channle_gain_based_data_rate[i] < self.rate_threshold:
                        idx = np.where(self.channel_gain[i] == np.max(self.channel_gain[i]))[0][0]
                        interference = 0
                        for j in range(self.SAT_len * self.SAT_plane):
                            if idx == j: continue
                            elif self.coverage_indicator[i][j] == 1 and channle_based_load_info[j] > 0:
                                    interference += self.SAT_Tx_power*self.anttena_gain*self.channel_gain[i][j]
                        FSPL = (299792458/(4*np.pi*self.freq*1000000*np.linalg.norm(self.GS[i] - self.SAT_point[idx]))) ** 2
                        data_rate = self.SAT_BW/channle_based_load_info[idx] * np.log2(1 + (self.SAT_Tx_power*self.anttena_gain*FSPL*self.channel_gain[i][idx] / (interference*FSPL + 10**(-12))))
                        

                        # HOF: non-coverage area
                        if self.coverage_indicator[i][idx] == 0:
                            self.channel_based_status_log[i][self.timestep] = 1
                        # HOF: QoS
                        elif data_rate < self.rate_threshold:
                            self.channel_based_status_log[i][self.timestep] = 2
                        # HO
                        else:
                            self.channel_based_service_index[i] = idx
                            self.channel_based_status_log[i][self.timestep] = 3
                    else:
                        self.channel_based_status_log[i][self.timestep] = 4                        
                
                if _actions[i] == -1 : continue # non-coverage 건너뛰기
                reward = 0
                self.load_log[i][self.timestep] = np.count_nonzero(_actions == _actions[i])
                # HO occur
                if _service_indicator[i][self.states[self.agents[i]]] == 0:                    
                    # HOF: data rate 
                    if self.data_rate[i] < self.rate_threshold:
                        reward = -30
                        self.agent_status_log[i][self.timestep] = 2
                        self.service_indicator[i] = np.zeros(self.SAT_len*self.SAT_plane) # 다음 time slot에 무조건 HO가 일어나도록 설정; 대기 상태
                    # HO cost
                    else:
                        reward = -15
                        self.agent_status_log[i][self.timestep] = 3
                # Ack
                else:
                    if self.data_rate[i] < self.rate_threshold:
                        reward = -30
                        self.agent_status_log[i][self.timestep] = 2
                        self.service_indicator[i] = np.zeros(self.SAT_len*self.SAT_plane) # 다음 time slot에 무조건 HO가 일어나도록 설정; 대기 상태
                    else:
                        reward = self.visible_time_weight * self.visible_time[i][_actions[i]] + self.rate_weight * self.data_rate[i]
                        self.agent_status_log[i][self.timestep] = 4
                        if self.debugging: print(f"ACK Status, {i}-th GS, Selected SAT: {_actions[i]}, load: {np.count_nonzero(_actions == _actions[i])}, Data rate: {self.data_rate[i]}")
                self.rewards[self.agents[i]] = reward

            if self.render_mode == "human":
                self.render()

            # Check termination conditions
            if self.timestep == self.terminal_time:
                self.terminations = {agent: True for agent in self.agents}
                if self.debugging:
                    self.render_result()
            else:
                self.timestep += 1
        else:
            self._clear_rewards()
        
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        #return self.observations, self.rewards, self.terminations, self.truncations, self.infos
        
    def render(self):
        """
        Rendering scenario step

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
        # Plot Selected line; 대기 상태면 선 x
        for i in range(self.GS_size):
            if np.count_nonzero(self.service_indicator[i]):
                axes.plot(
                    (self.GS[i,0], self.SAT_point[self.states[self.agents[i]],0]),
                    (self.GS[i,1], self.SAT_point[self.states[self.agents[i]],1]),
                    "--k", linewidth=1
                    )

        axes.set_aspect(1)
        axes.axis([-50, 200, -50, 150])

        plt.show()
    
    def render_result(self):
        """
        Plot result graphs

        0. Agents' Handover Average
        1. MVT
        2. MAC
        3. SINR-based
        """

        # data rate 평균 (0포함) 수정해야함 -> Ack 시 통신 rate만 추출해야함
        for i in range(self.GS_size):
            MADL_rate = 0
            MVT_rate = 0
            random_rate = 0
            CG_based_rate = 0
            print(f"{i}-th Agent's episode average data rate (MADL based): {np.average(self.rate_data_log[i,:])}")
            MADL_rate += np.average(self.rate_data_log[i,:])
            print(f"{i}-th Agent's episode average data rate (MVT based): {np.average(self.MVT_data_rate_log[i,:])}")
            MVT_rate += np.average(self.MVT_data_rate_log[i,:])
            print(f"{i}-th Agent's episode average data rate (random based): {np.average(self.random_data_rate_log[i,:])}")
            random_rate += np.average(self.random_data_rate_log[i,:])
            print(f"{i}-th Agent's episode average data rate (CG based): {np.average(self.channel_based_data_rate_log[i,:])}")
            CG_based_rate += np.average(self.channel_based_data_rate_log[i,:])
        print(f"MADL episode average SINR:{MADL_rate/self.GS_size}")
        print(f"MVT episode average data rate:{MVT_rate/self.GS_size}")
        print(f"RANDOM episode average data rate:{random_rate/self.GS_size}")
        print(f"CG-based episode average data rate:{CG_based_rate/self.GS_size}")

        # Plot Agents' Status
        plt.figure(1)
        agents = np.arange(0,10)
        _status = np.zeros((self.GS_size, 5))
        for i in range(self.GS_size):
            _status[i][0] = np.count_nonzero(self.agent_status_log[i] == 1)
            _status[i][1] = np.count_nonzero(self.agent_status_log[i] == 2)
            _status[i][2] = np.count_nonzero(self.agent_status_log[i] == 3)
            _status[i][3] = np.count_nonzero(self.agent_status_log[i] == 4)

        bar_width = 0.1
        status_1 = plt.bar(agents, _status[:,0], bar_width, label='HOF-non_service')
        status_2 = plt.bar(agents + bar_width, _status[:,1], bar_width, label='HOF-QoS')
        status_3 = plt.bar(agents + 2*bar_width, _status[:,2], bar_width, label='HO')
        status_4 = plt.bar(agents + 3*bar_width, _status[:,3], bar_width, label='ACK')
        plt.xticks(np.arange(bar_width, 10+bar_width,1), agents)
        plt.xlabel('# of Agent'); plt.legend(); plt.title("MADL-based")

        # Plot Benchmark-MVT
        plt.figure(2)
        MVT_status = np.zeros((self.GS_size, 5))
        for i in range(self.GS_size):
            MVT_status[i][0] = np.count_nonzero(self.MVT_status_log[i] == 1)
            MVT_status[i][1] = np.count_nonzero(self.MVT_status_log[i] == 2)
            MVT_status[i][2] = np.count_nonzero(self.MVT_status_log[i] == 3)
            MVT_status[i][3] = np.count_nonzero(self.MVT_status_log[i] == 4)

        bar_width = 0.1
        status_1 = plt.bar(agents, MVT_status[:,0], bar_width, label='HOF-non_service')
        status_2 = plt.bar(agents + bar_width, MVT_status[:,1], bar_width, label='HOF-QoS')
        status_3 = plt.bar(agents + 2*bar_width, MVT_status[:,2], bar_width, label='HO')
        status_4 = plt.bar(agents + 3*bar_width, MVT_status[:,3], bar_width, label='ACK')
        plt.xticks(np.arange(bar_width, 10+bar_width,1), agents)
        plt.xlabel('# of Agent'); plt.legend(); plt.title("MVT")

        # Plot Benchmark-Random
        plt.figure(3)
        Random_status = np.zeros((self.GS_size, 5))
        for i in range(self.GS_size):
            Random_status[i][0] = np.count_nonzero(self.random_status_log[i] == 1)
            Random_status[i][1] = np.count_nonzero(self.random_status_log[i] == 2)
            Random_status[i][2] = np.count_nonzero(self.random_status_log[i] == 3)
            Random_status[i][3] = np.count_nonzero(self.random_status_log[i] == 4)

        bar_width = 0.1
        status_1 = plt.bar(agents, Random_status[:,0], bar_width, label='HOF-non_service')
        status_2 = plt.bar(agents + bar_width, Random_status[:,1], bar_width, label='HOF-QoS')
        status_3 = plt.bar(agents + 2*bar_width, Random_status[:,2], bar_width, label='HO')
        status_4 = plt.bar(agents + 3*bar_width, Random_status[:,3], bar_width, label='ACK')
        plt.xticks(np.arange(bar_width, 10+bar_width,1), agents)
        plt.xlabel('# of Agent'); plt.legend(); plt.title("Random")

        # Plot Benchmark-channel-based
        plt.figure(4)
        channel_based_status = np.zeros((self.GS_size, 5))
        for i in range(self.GS_size):
            channel_based_status[i][0] = np.count_nonzero(self.channel_based_status_log[i] == 1)
            channel_based_status[i][1] = np.count_nonzero(self.channel_based_status_log[i] == 2)
            channel_based_status[i][2] = np.count_nonzero(self.channel_based_status_log[i] == 3)
            channel_based_status[i][3] = np.count_nonzero(self.channel_based_status_log[i] == 4)            

        bar_width = 0.1
        status_1 = plt.bar(agents, channel_based_status[:,0], bar_width, label='HOF-non_service')
        status_2 = plt.bar(agents + bar_width, channel_based_status[:,1], bar_width, label='HOF-QoS')
        status_3 = plt.bar(agents + 2*bar_width, channel_based_status[:,2], bar_width, label='HO')
        status_4 = plt.bar(agents + 3*bar_width, channel_based_status[:,3], bar_width, label='ACK')        
        plt.xticks(np.arange(bar_width, 10+bar_width,1), agents)
        plt.xlabel('# of Agent'); plt.legend(); plt.title("Channel-gain-based")

        # ACK variance
        print(f"ACK var --> MADQN: {np.var(_status[:,3])}, MVT: {np.var(MVT_status[:,3])}, Random: {np.var(Random_status[:,3])}, SINR: {channel_based_status[:,3]}")
        
        # Plot comparsion of Ack HO strategies
        plt.figure(7)
        bar_width = 0.1
        status_1 = plt.bar(agents, _status[:,3], bar_width, label='MADQN')
        status_2 = plt.bar(agents + bar_width, MVT_status[:,3], bar_width, label='MVT')
        status_3 = plt.bar(agents + 2*bar_width, Random_status[:,3], bar_width, label='Random')
        status_4 = plt.bar(agents + 3*bar_width, channel_based_status[:,3], bar_width, label='MAX-CG')
        plt.xticks(np.arange(bar_width, 10+bar_width,1), agents)
        plt.xlabel('UEs'); plt.legend(); plt.ylabel('Communication times')

        # Plot comparsion of average handover
        plt.figure(5)
        HO_MADQN = np.zeros((self.terminal_time+1))
        HO_MVT   = np.zeros((self.terminal_time+1))
        HO_RANDOM   = np.zeros((self.terminal_time+1))
        HO_CG = np.zeros((self.terminal_time+1))
        for agent in range(self.GS_size):
            for t in range(self.terminal_time+1):
                if self.agent_status_log[agent][t] == 3:
                    HO_MADQN[t:] += 1
                if self.MVT_status_log[agent][t] == 3:
                    HO_MVT[t:] += 1
                if self.random_status_log[agent][t] == 3:
                    HO_RANDOM[t:] += 1
                if self.channel_based_status_log[agent][t] == 3:
                    HO_CG[t:] += 1
        
        HO_MADQN[:] /= self.GS_size
        HO_MVT[:] /= self.GS_size
        HO_RANDOM[:] /= self.GS_size
        HO_CG[:] /= self.GS_size
        time_step = np.arange(self.terminal_time)
        interval = 15 # mark interveal

        plt.plot(time_step, HO_MADQN[1:], label='MADQN', marker='*', markevery=interval)

        plt.plot(time_step, HO_MVT[1:], label='MVT', marker='.', markevery=interval)        

        plt.plot(time_step, HO_RANDOM[1:], label='Random', marker='|', markevery=interval)

        plt.plot(time_step, HO_CG[1:], label='MAX-CG', marker='P', markevery=interval)

        plt.xlim((1,155))
        plt.ylabel('Average handover'); plt.legend(); plt.xlabel('time step'); plt.grid()


        # Plot comprasion of communication failure rate
        plt.figure(6)
        HOF_MADQN = np.zeros((self.terminal_time+1))
        HOF_MVT   = np.zeros((self.terminal_time+1))
        HOF_RANDOM   = np.zeros((self.terminal_time+1))
        HOF_CG  = np.zeros((self.terminal_time+1))
        for agent in range(self.GS_size):
            for t in range(self.terminal_time+1):
                if self.agent_status_log[agent][t] == 1 or self.agent_status_log[agent][t] == 2:
                    HOF_MADQN[t:] += 1
                if self.MVT_status_log[agent][t] == 1 or self.MVT_status_log[agent][t] == 2:
                    HOF_MVT[t:] += 1
                if self.random_status_log[agent][t] == 1 or self.random_status_log[agent][t] == 2:
                    HOF_RANDOM[t:] += 1
                if self.channel_based_status_log[agent][t] == 1 or self.channel_based_status_log[agent][t] == 2:
                    HOF_CG[t:] += 1

        HOF_MADQN /= self.GS_size
        HOF_MVT   /= self.GS_size
        HOF_RANDOM   /= self.GS_size
        HOF_CG  /= self.GS_size

        _HOF_MADQN = (HOF_MADQN[-1]) / self.terminal_time
        _HOF_MVT   = (HOF_MVT[-1]) / self.terminal_time
        _HOF_RANDOM   = (HOF_RANDOM[-1]) / self.terminal_time
        _HOF_CG  = (HOF_CG[-1]) / self.terminal_time
        
        HOF_MADQN[:] = _HOF_MADQN
        HOF_MVT[:]   = _HOF_MVT  
        HOF_RANDOM[:]   = _HOF_RANDOM  
        HOF_CG[:]  = _HOF_CG 

        time_step = np.arange(self.terminal_time)
        interval = 15 # mark interveal

        plt.plot(time_step, HOF_MADQN[1:], label='MADQN', marker='*', markevery=interval)

        plt.plot(time_step, HOF_MVT[1:], label='MVT', marker='.', markevery=interval)        

        plt.plot(time_step, HOF_RANDOM[1:], label='Random', marker='|', markevery=interval)

        plt.plot(time_step, HOF_CG[1:], label='MAX-CG', marker='P', markevery=interval)

        plt.xlim((1,155))
        plt.ylabel("Average communication failure rate"); plt.legend(loc=(0.02, 0.5)); plt.xlabel('time step'); plt.grid()

        # Data rate log
        plt.figure(8)
        time_step = np.arange(self.terminal_time+1)
        for i in range(self.GS_size):
            plt.step(time_step, self.rate_data_log[i])
        plt.title("Agents' Data rate log")

        print(f"MADQN average HO: {HO_MADQN[-1]}")
        print(f"MVT average HO: {HO_MVT[-1]}")
        print(f"RANDOM average HO: {HO_RANDOM[-1]}")
        print(f"MAX CG average HO: {HO_CG[-1]}")

        print(f"MADQN average HOF rate: {HOF_MADQN[-1]}")
        print(f"MVT average HOF rate: {HOF_MVT[-1]}")
        print(f"RANDOM average HOF rate: {HOF_RANDOM[-1]}")
        print(f"MAX CG average HOF rate: {HOF_CG[-1]}")

        cnt_HOF_MADQN = np.sum(_status[:,2])
        cnt_HOF_MVT   = np.sum(MVT_status[:,2])
        cnt_HOF_RANDOM   = np.sum(Random_status[:,2])
        cnt_HOF_CG  = np.sum(channel_based_status[:,2])

        print(f"MADQN 스킴 핸드오버실패 총 횟수: {cnt_HOF_MADQN},\
              QoS: 횟수-{np.sum(_status[:,2])}, 비율-{(np.sum(_status[:,2]))/cnt_HOF_MADQN}")
        print(f"MVT 스킴 핸드오버실패 총 횟수: {cnt_HOF_MVT}, \
              QoS: 횟수-{np.sum(MVT_status[:,2])}, 비율-{(np.sum(MVT_status[:,2]))/cnt_HOF_MVT}")
        print(f"RANDOM 스킴 핸드오버실패 총 횟수: {cnt_HOF_RANDOM}, \
              QoS: 횟수-{np.sum(Random_status[:,2])}, 비율-{(np.sum(Random_status[:,2]))/cnt_HOF_RANDOM}")
        print(f"MAX-CG 스킴 핸드오버실패 총 횟수: {cnt_HOF_CG},  \
              QoS: 횟수-{np.sum(channel_based_status[:,2])}, 비율-{(np.sum(channel_based_status[:,2]))/cnt_HOF_CG}")

        plt.show()


