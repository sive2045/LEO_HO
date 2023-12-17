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
        
        self.rate_threshold = 1_000_000 # bps
        self.rate_data_log = np.zeros((self.GS_size, self.terminal_time + 1))

        self.HO_log = np.zeros((self.GS_size, self.terminal_time + 1))

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
        self.SAT_Load = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane)) # the available channels of SAT
        self.load_info = np.zeros((self.GS_size, self.SAT_len*self.SAT_plane)) # load info
        
        self.SAT_BW = 8_000_000_000 # Hz BW budget of SAT
        self.freq = 20 # GHz
        self.SAT_Tx_power = 40 # W
        self.GNSS_noise = 1 # 수정 예정
        self.anttena_gain = 1_000
        
        self.visible_time_weight = 0.5
        self.rate_weight = 2*(10**(-7))

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
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1e13, shape=(4, self.SAT_len * self.SAT_plane), dtype=np.float64
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.SAT_len * self.SAT_plane,), dtype=np.int8
                    ),
                }
            )
            for name in self.agents
        }
        # self.observation_spaces = {
        #     i: spaces.Box(
        #                 low=0, high=1e13, shape=(3, self.SAT_len * self.SAT_plane), dtype=np.float64
        #         )
        #     for i in self.agents
        # }
        # self.observation_spaces = {
        #     i: spaces.Dict(
        #         {
        #             "observation": spaces.Box(
        #                 low=0, high=1e13, shape=(3, self.SAT_len * self.SAT_plane), dtype=np.float64
        #             ),
        #         }
        #     )
        #     for i in self.agents
        #}
        #|------Debugging args-----------------------------------------------------------------------------------------------------------------------|
        self.render_mode = render_mode        
        self.debugging = debugging
        
        # Result vars
        # Ack, Blocked, HOF
        # --> non-coverage는 어떻게 뺄지 고민해야함.
        self.agent_status_log = np.zeros((self.GS_size, self.terminal_time+1)) # 1: non-serviced, 2: HOF-QoS, 3: HOC, 4: ACK
        self.SINR_log = np.zeros((self.GS_size, self.terminal_time+1))
        self.load_log = np.zeros((self.GS_size, self.terminal_time+1))

    
    def observation_space(self, agent):
        return self.observation_spaces[agent]['observation']

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
        self.MVT_status_log = np.zeros((self.GS_size, self.terminal_time+1)) # 1: non-serviced, 2: HOF-QoS, 3: HOC, 4: ACK
        self.MVT_service_index = np.zeros((self.GS_size)) # SAT index
        self.MVT_service_index = np.random.randint(0, self.SAT_len*self.SAT_plane, (self.SAT_len*self.SAT_plane)) # init index
        self.MVT_data_rate_log = np.zeros((self.GS_size, self.terminal_time+1))
        self.MVT_SAT_Load_MAX = np.full(self.SAT_len*self.SAT_plane, 50) # the maximum available channels of SAT
        self.MVT_SAT_Load = np.zeros((self.SAT_len * self.SAT_plane)) # the available channels of SAT
        self.MVT_HO_log = np.zeros((self.GS_size, self.terminal_time + 1))
        
        # random
        self.random_status_log = np.zeros((self.GS_size, self.terminal_time+1)) # 1: non-serviced, 2: HOF-QoS, 3: HOC, 4: ACK
        self.random_service_index = np.zeros((self.GS_size)) # SAT index
        self.random_service_index = np.random.randint(0, self.SAT_len*self.SAT_plane, self.SAT_len*self.SAT_plane) # init index
        self.random_data_rate_log = np.zeros((self.GS_size, self.terminal_time+1))
        self.random_SAT_Load_MAX = np.full(self.SAT_len*self.SAT_plane, 50) # the maximum available channels of SAT
        self.random_SAT_Load = np.zeros((self.SAT_len * self.SAT_plane)) # the available channels of SAT
        self.ramdon_HO_log = np.zeros((self.GS_size, self.terminal_time + 1))

        # channel-gain-based
        self.channel_based_status_log = np.zeros((self.GS_size, self.terminal_time+1)) # 1: non-serviced, 2: HOF-QoS, 3: HOC, 4: ACK
        self.channel_based_service_index = np.zeros((self.GS_size)) # SAT index
        self.channel_based_service_index = np.random.randint(0, self.SAT_len*self.SAT_plane, self.SAT_len*self.SAT_plane) # init index
        self.channel_based_data_rate_log = np.zeros((self.GS_size, self.terminal_time+1))
        self.channel_gaind_based_SAT_Load_MAX = np.full(self.SAT_len*self.SAT_plane, 50) # the maximum available channels of SAT
        self.channel_gaind_based_SAT_Load = np.zeros((self.SAT_len * self.SAT_plane)) # the available channels of SAT
        self.CG_HO_log = np.zeros((self.GS_size, self.terminal_time + 1))

        # max-available channel based
        self.max_available_channel_based_status_log = np.zeros((self.GS_size, self.terminal_time+1)) # 1: non-serviced, 2: HOF-QoS, 3: HOC, 4: ACK
        self.max_available_channel_based_service_index = np.zeros((self.GS_size)) # SAT index
        self.max_available_channel_based_service_index = np.random.randint(0, self.SAT_len*self.SAT_plane, self.SAT_len*self.SAT_plane) # init index
        self.max_available_channel_based_data_rate_log = np.zeros((self.GS_size, self.terminal_time+1))
        self.max_available_channel_gaind_based_SAT_Load_MAX = np.full(self.SAT_len*self.SAT_plane, 50) # the maximum available channels of SAT
        self.max_available_channel_gaind_based_SAT_Load = np.zeros((self.SAT_len * self.SAT_plane)) # the available channels of SAT
        self.MAC_HO_log = np.zeros((self.GS_size, self.terminal_time + 1))

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
        return coverage_indicator        

    def _get_visible_time(self):
        """
        return visible time btw SAT and GS
        """
        self.visible_time = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane))
        for i in range(self.GS_size):
            for j in range(self.SAT_len * self.SAT_plane):
                if self.coverage_indicator[i][j] == 1:
                    dist = np.sqrt( (self.GS[i][0]-self.SAT_point[j][0])**2 + (self.GS[i][1]-self.SAT_point[j][1])**2 )
                    self.visible_time[i][j] = dist / self.SAT_speed     

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
        #if self.debugging: print(f"{self.timestep}-times Agents' channel gain: {self.channel_gain}")
        return self.channel_gain

    def _cal_data_rate(self, SAT_load):
        data_rate = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane), dtype='float64')
        noise_0 = 4*(10**(-21))
        for i in range(self.GS_size):
            interference = 1
            for j in range(self.SAT_len * self.SAT_plane):
                if self.coverage_indicator[i][j] == 1:
                    FSPL = (299792458/(4*np.pi*self.freq*1000000*np.linalg.norm(self.GS[i] - self.SAT_point[j]))) ** 2
                    interference += self.SAT_Tx_power*self.anttena_gain*self.channel_gain[i][j]*FSPL
            
            for t in range(self.SAT_len * self.SAT_plane):
                if self.coverage_indicator[i][t]:
                    FSPL = (299792458/(4*np.pi*self.freq*1000000*np.linalg.norm(self.GS[i] - self.SAT_point[t]))) ** 2
                    signal_power = self.SAT_Tx_power*self.anttena_gain*FSPL*self.channel_gain[i][t]
                    if SAT_load[i][t] > 0:
                        data_rate[i][t] = self.SAT_BW/SAT_load[i][t] * np.log2(1 + (signal_power / ((interference-signal_power) + (self.SAT_BW/SAT_load[i][t])*noise_0 )) )
                    else:
                        data_rate[i][t] = self.SAT_BW * np.log2(1 + (signal_power / ((interference-signal_power) + (self.SAT_BW)*noise_0 )) )

        #if self.debugging: print(f"{self.timestep}-times Agents' data rate: {data_rate}")
        return data_rate

    def _update_load_SAT(self, actions) -> None:
        '''
        time slot 한번당 실행
        -> 초기화 후 사용
        '''
        self.SAT_Load = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane))
        for i in range(self.GS_size):
            self.SAT_Load[:, actions[i]] += 1
    
    def _update_benchmark_load_info(self, actions):
        '''
        벤치마킹 load 정보 업데이트
        '''
        SAT_Load = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane))
        for i in range(self.GS_size):            
            SAT_Load[:, int(actions[i])] += 1
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
        self._get_visible_time()
        # Update channel gain
        self.channel_gain = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane))
        # Update data rate
        self.data_rate = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane), dtype='float64')
        # init load info
        self.SAT_Load = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane))

        # observations
        self.observations = {
            f"groud_station_{i}": {
                'observation': np.zeros((4, self.SAT_len * self.SAT_plane)),
                'action_mask': np.zeros((self.SAT_len * self.SAT_plane))
            } for i in range(self.GS_size)
        }
        for i in range(self.GS_size):
            observation = [
                self.coverage_indicator[i],
                self.visible_time[i],
                self.channel_gain[i],
                self.data_rate[i]
            ]
            self.observations[self.agents[i]]['observation'] = observation
        
        # logs
        self.agent_status_log = np.zeros((self.GS_size, self.terminal_time+1)) # 1: non-serviced, 2: HOF-QoS, 3: HO, 4: ACK 

        # Benchmark Scheme
        if self.debugging:
            self.init_benchmark_scheme()

        #return self.observations, None
    
    def observe(self, agent):
        gs_idx = int(agent.split('_')[-1])
        
        if gs_idx == 0:
            '''
            coverage indicator update는 여기서 이뤄짐
            agent_#->observe agent_#->,,, 순이라
            last agent일때 업데이트 하면 해당 coverage가 반영이 안됨 << action mask 관련 설정임
            '''
            # Update SAT position
            self.SAT_point = self._SAT_coordinate(self.SAT_point, self.SAT_len, self.timestep, self.SAT_speed)
            # Update GS position
            self.GS = self._GS_random_walk(self.GS, self.GS_speed)        
            # Update coverage indicator
            self.coverage_indicator = self._is_in_coverage(self.SAT_point, self.GS, self.SAT_coverage_radius)     
            # Update channel gain
            self._cal_shadowed_rice_fading_gain() 
            # visible time
            self._get_visible_time()

        ## coverage 내부에서만 선택하도록 만듦
        action_mask = np.zeros((self.SAT_len*self.SAT_plane))
        # tmp data rate --> load가 없을 경우의 data rate를 계산해서 해당 data rate가 임계값 이상인 경우에만 선택 가능하도록 설정
        # 최소한의 방지 역할
        interference = 1
        for j in range(self.SAT_len * self.SAT_plane):
            if self.coverage_indicator[gs_idx][j] == 1:
                FSPL = (299792458/(4*np.pi*self.freq*1000000*np.linalg.norm(self.GS[gs_idx] - self.SAT_point[j]))) ** 2
                interference += self.SAT_Tx_power*self.anttena_gain*self.channel_gain[gs_idx][j]*FSPL

        for i in range(self.SAT_len*self.SAT_plane):
            if self.coverage_indicator[gs_idx][i] == 1:
                FSPL = (299792458/(4*np.pi*self.freq*1000000*np.linalg.norm(self.GS[gs_idx] - self.SAT_point[i]))) ** 2
                signal_power = self.SAT_Tx_power*self.anttena_gain*FSPL*self.channel_gain[gs_idx][i]
                tmp_data_rate = self.SAT_BW * np.log2(1 + (signal_power / ((interference-signal_power) + (self.SAT_BW)* 4*(10**(-21)) )) )
                if tmp_data_rate >= self.rate_threshold :
                    action_mask[i] = 1
        # if self.debugging: print(f"{self.timestep}-th, gsidx: {gs_idx} observe func !! {action_mask}")
        self.observations[agent]['action_mask'] = action_mask
        return {"observation": self.observations[agent]['observation'], "action_mask": action_mask}

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
        self.states[self.agent_selection] = action
        if self.debugging: 
            print(f"timestep:{self.timestep}, agent: {agent}") 
            print(f"timestep:{self.timestep}, state : {self.states[agent]}")
        
        if self._agent_selector.is_last():
            # Update service indicator
            _service_indicator = np.copy(self.service_indicator)
            self.service_indicator = np.zeros((self.GS_size, self.SAT_len*self.SAT_plane))
            for i in range(self.GS_size):
                self.service_indicator[i][self.states[self.agents[i]]] = 1
            # # Update SAT position
            # self.SAT_point = self._SAT_coordinate(self.SAT_point, self.SAT_len, self.timestep, self.SAT_speed)
            # # Update GS position
            # self.GS = self._GS_random_walk(self.GS, self.GS_speed)
            # if self.debugging:  print(f"timestep:{self.timestep}, agent poistion: {self.GS}")            
            # # Update coverage indicator
            # self.coverage_indicator = self._is_in_coverage(self.SAT_point, self.GS, self.SAT_coverage_radius)
            # if self.debugging: print(f"coverage indicator {self.coverage_indicator}")            
            # Update load info
            _actions = np.array(list(self.states.values()))
            self._update_load_SAT(_actions)
            # Update channel gain
            # self._cal_shadowed_rice_fading_gain()
            # Update Data rate
            self.data_rate = self._cal_data_rate(self.SAT_Load)

            for i in range(self.GS_size):
                observation = (
                    self.coverage_indicator[i],
                    self.visible_time[i],
                    self.channel_gain[i],  
                    self.data_rate[i]
                )
                self.observations[f"groud_station_{i}"]['observation'] = observation
                # if self.debugging:
                #     print(f'observation!! :{self.observations[f"groud_station_{i}"]}')
                #     print(f"actions: {self.action_spaces[f'groud_station_{i}']}")
                # 선택한 위성의 data rate log 저장
                #self.rate_data_log[i, self.timestep] =  self.data_rate[i, _actions[i]]
                # HO시도 -> HO 횟수로 취급
                if (_service_indicator[i] == self.service_indicator[i]).min() == False:
                    self.HO_log[i][self.timestep] = 1

            # update benchmark data rate
            if self.debugging:
                '''
                data rate 수정해야함!
                idx 먼저 고르고
                그 이후에 data rate 계산!
                '''
                MVT_data_rate = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane))
                random_data_rate = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane))
                channle_gain_based_data_rate = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane))
                max_available_channel_data_rate = np.zeros((self.GS_size, self.SAT_len * self.SAT_plane))

                # selected index
                MVT_tmp_index = np.zeros((self.GS_size), dtype=int)
                MAX_CG_index = np.zeros((self.GS_size), dtype=int)
                random_tmp_index = np.zeros((self.GS_size), dtype=int)
                for i in range(self.GS_size):
                    MVT_tmp_index[i] = int(np.where(self.visible_time[i] == np.max(self.visible_time[i]))[0][0])
                    MAX_CG_index[i] = int(np.where(self.channel_gain[i] == np.max(self.channel_gain[i]))[0][0])
                    rnd_tmp_index_array = np.where(self.coverage_indicator[i] == 1)
                    random_tmp_index[i] = int(np.random.choice(rnd_tmp_index_array[0], size=1))

                # get load info
                MVT_load_info = self._update_benchmark_load_info(MVT_tmp_index)
                random_load_info = self._update_benchmark_load_info(random_tmp_index)
                channle_based_load_info = self._update_benchmark_load_info(MAX_CG_index)
                max_available_channel_load_info = self._update_benchmark_load_info(self.max_available_channel_based_service_index)
                # print(f"MVT_load_info: {MVT_load_info}")
                # print(f"random_load_info: {random_load_info}")
                # print(f"channle_based_load_info: {channle_based_load_info}")
                # print(f"max_available_channel_load_info: {max_available_channel_load_info}")

                # get data rate
                MVT_data_rate = self._cal_data_rate(MVT_load_info)
                random_data_rate = self._cal_data_rate(random_load_info)
                channle_gain_based_data_rate = self._cal_data_rate(channle_based_load_info)
                max_available_channel_data_rate = self._cal_data_rate(max_available_channel_load_info)

            # rewards
            for i in range(self.GS_size):
                # Benchmark
                if self.debugging:                     
                    print(f"{self.timestep}-time step, {i}-th agent info: MVT: {self.MVT_service_index[i]}")
                    print(f"{self.timestep}-time step, {i}-th agent info: RANDOM: {self.random_service_index[i]}")
                    print(f"{self.timestep}-time step, {i}-th agent info: channel-gain-based: {self.channel_based_service_index[i]}")
                    # MVT                                    
                    if MVT_tmp_index[i] !=  self.MVT_service_index[i]: 
                        # HO시도 -> HO 횟수로 취급
                        self.MVT_HO_log[i][self.timestep] = 1                    
                        # HOF: QoS
                        if MVT_data_rate[i, MVT_tmp_index[i]] < self.rate_threshold:
                            self.MVT_service_index[i] = -1
                            self.MVT_status_log[i][self.timestep] = 2
                            self.MVT_data_rate_log[i, self.timestep] = 0
                        # HO
                        else:
                            self.MVT_service_index[i] = MVT_tmp_index[i]
                            self.MVT_status_log[i][self.timestep] = 3
                            self.MVT_data_rate_log[i, self.timestep] = 0   
                    else:
                        # Comm F: QoS
                        if MVT_data_rate[i, MVT_tmp_index[i]] < self.rate_threshold:
                            self.MVT_service_index[i] = -1
                            self.MVT_status_log[i][self.timestep] = 2
                            self.MVT_data_rate_log[i, self.timestep] = 0
                        else: 
                            self.MVT_status_log[i][self.timestep] = 4
                            self.MVT_data_rate_log[i, self.timestep] = MVT_data_rate[i, self.MVT_service_index[i]]   
                    
                    # RANDOM               
                    if random_tmp_index[i] !=  self.random_service_index[i]: 
                        # HO시도 -> HO 횟수로 취급
                        self.ramdon_HO_log[i][self.timestep] = 1
                    
                        if self.coverage_indicator[i][random_tmp_index[i]] == 0:
                            self.random_service_index[i] = -1
                            self.random_data_rate_log[i, self.timestep] = 0
                            self.random_status_log[i][self.timestep] = 1
                        elif random_data_rate[i, random_tmp_index[i]] < self.rate_threshold:
                            self.random_service_index[i] = -1
                            self.random_status_log[i][self.timestep] = 2
                            self.random_data_rate_log[i, self.timestep] = 0
                        # HO
                        else:
                            self.random_service_index[i] = random_tmp_index[i]
                            self.random_status_log[i][self.timestep] = 3
                            self.random_data_rate_log[i, self.timestep] = 0
                    else:
                        if random_data_rate[i, random_tmp_index[i]] < self.rate_threshold:
                            self.random_service_index[i] = -1
                            self.random_status_log[i][self.timestep] = 2
                            self.random_data_rate_log[i, self.timestep] = 0
                        else:
                            self.random_status_log[i][self.timestep] = 4
                            self.random_data_rate_log[i, self.timestep] = random_data_rate[i, self.random_service_index[i]]

                    # channel gain-based                   
                    if  MAX_CG_index[i] !=  self.channel_based_service_index[i]: 
                        # HO시도 -> HO 횟수로 취급
                        self.CG_HO_log[i][self.timestep] = 1
                        
                        # HOF: non-coverage area
                        if self.coverage_indicator[i][ MAX_CG_index[i]] == 0:
                            self.channel_based_service_index[i] = -1
                            self.channel_based_status_log[i][self.timestep] = 1
                            self.channel_based_data_rate_log[i, self.timestep] = 0
                        # HOF: QoS
                        elif channle_gain_based_data_rate[i,  MAX_CG_index[i]] < self.rate_threshold:
                            self.channel_based_service_index[i] = -1
                            self.channel_based_status_log[i][self.timestep] = 2
                            self.channel_based_data_rate_log[i, self.timestep] = 0
                        # HO
                        else:
                            self.channel_based_service_index[i] =  MAX_CG_index[i]
                            self.channel_based_status_log[i][self.timestep] = 3
                            self.channel_based_data_rate_log[i, self.timestep] = 0
                    else:
                        # HOF: QoS
                        if channle_gain_based_data_rate[i,  MAX_CG_index[i]] < self.rate_threshold:
                            self.channel_based_service_index[i] = -1
                            self.channel_based_status_log[i][self.timestep] = 2
                            self.channel_based_data_rate_log[i, self.timestep] = 0
                        else:
                            self.channel_based_status_log[i][self.timestep] = 4  
                            self.channel_based_data_rate_log[i, self.timestep] = channle_gain_based_data_rate[i, self.channel_based_service_index[i]]                      

                    # MAX-available channel
                    # 두번 돌려야 함 -> data rate 얻기가 애매함
                    # idx = -100
                    # tmp_channel_num = 50
                    # for j in range(self.SAT_len * self.SAT_plane):
                    #     if self.coverage_indicator[i][j] == 1 and tmp_channel_num >= max_available_channel_load_info[i, j]:                            
                    #         idx = j
                    #         tmp_channel_num = max_available_channel_load_info[i, j]
                    # if idx !=  self.max_available_channel_based_service_index[i]: 
                    #     # HO시도 -> HO 횟수로 취급
                    #     self.MAC_HO_log[i][self.timestep] = 1
                        
                    #     # HOF: non-coverage area
                    #     if self.coverage_indicator[i][idx] == 0:
                    #         self.max_available_channel_based_service_index[i] = -1
                    #         self.max_available_channel_based_status_log[i][self.timestep] = 1
                    #         self.max_available_channel_based_data_rate_log[i, self.timestep] = 0
                    #     # HOF: QoS
                    #     elif max_available_channel_data_rate[i, idx] < self.rate_threshold:
                    #         self.max_available_channel_based_service_index[i] = -1
                    #         self.max_available_channel_based_status_log[i][self.timestep] = 2
                    #         self.max_available_channel_based_data_rate_log[i, self.timestep] = 0
                    #     # HO
                    #     else:
                    #         self.max_available_channel_based_service_index[i] = idx
                    #         self.max_available_channel_based_status_log[i][self.timestep] = 3
                    #         self.max_available_channel_based_data_rate_log[i, self.timestep] = 0

                    # else:
                    #     # HOF: QoS
                    #     if max_available_channel_data_rate[i, idx] < self.rate_threshold:
                    #         self.max_available_channel_based_service_index[i] = -1
                    #         self.max_available_channel_based_status_log[i][self.timestep] = 2
                    #         self.max_available_channel_based_data_rate_log[i, self.timestep] = 0
                    #     else:
                    #         self.max_available_channel_based_status_log[i][self.timestep] = 4
                    #         self.max_available_channel_based_data_rate_log[i, self.timestep] = max_available_channel_data_rate[i, self.max_available_channel_based_service_index[i]]    

                reward = 0

                # HOF-non service SAT
                if self.coverage_indicator[i][_actions[i]] == 0:
                    reward = -15
                    self.agent_status_log[i][self.timestep] = 1
                    self.service_indicator[i] = np.zeros(self.SAT_len*self.SAT_plane) # 다음 time slot에 무조건 HO가 일어나도록 설정; 대기 상태
                    self.rewards[self.agents[i]] = reward
                    self.rate_data_log[i, self.timestep] =  0
                    #self.terminations = {agent: True for agent in self.agents} # 학습 종료.
                    print(f"agent:{i}-th, {self.timestep}, action: {_actions[i]}, coverage ; {self.coverage_indicator[i]}")
                # HO occur
                elif _service_indicator[i][_actions[i]] == 0:                    
                    # HOF: data rate 
                    if self.data_rate[i, _actions[i]] < self.rate_threshold:
                        reward = -15
                        self.agent_status_log[i][self.timestep] = 2
                        self.service_indicator[i] = np.zeros(self.SAT_len*self.SAT_plane) # 다음 time slot에 무조건 HO가 일어나도록 설정; 대기 상태
                        self.rewards[self.agents[i]] = reward
                        self.rate_data_log[i, self.timestep] = 0
                    # HO cost
                    else:
                        reward = -30
                        self.agent_status_log[i][self.timestep] = 3
                        self.rewards[self.agents[i]] = reward
                        self.rate_data_log[i, self.timestep] =  0
                # Ack
                else:
                    if self.data_rate[i, _actions[i]] < self.rate_threshold:
                        reward = -15
                        self.agent_status_log[i][self.timestep] = 2
                        self.service_indicator[i] = np.zeros(self.SAT_len*self.SAT_plane) # 다음 time slot에 무조건 HO가 일어나도록 설정; 대기 상태
                        self.rewards[self.agents[i]] = reward
                        self.rate_data_log[i, self.timestep] =  0
                    else:
                        reward = self.visible_time_weight * self.visible_time[i][_actions[i]] + np.min((50, self.rate_weight * self.data_rate[i][_actions[i]])) # data rate로 최대 reward는 10으로 설정 (100Mbps이상일때 너무 커지는 것을 방지)
                        self.agent_status_log[i][self.timestep] = 4
                        if self.debugging: print(f"ACK Status, {i}-th GS, Selected SAT: {_actions[i]}, load: {np.count_nonzero(_actions == _actions[i])}, Data rate: {self.data_rate[i][_actions[i]]}, Visble time {self.visible_time[i, _actions[i]]}, reawrd: {reward}")
                        self.rewards[self.agents[i]] = reward
                        self.rate_data_log[i, self.timestep] = self.data_rate[i, _actions[i]]

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
            MAX_C_based_rate = 0
            print(f"{i}-th Agent's episode average data rate (MADL based): {np.average(self.rate_data_log[i,:])}")
            MADL_rate += np.average(self.rate_data_log[i,:])
            print(f"{i}-th Agent's episode average data rate (MVT based): {np.average(self.MVT_data_rate_log[i,:])}")
            MVT_rate += np.average(self.MVT_data_rate_log[i,:])
            print(f"{i}-th Agent's episode average data rate (random based): {np.average(self.random_data_rate_log[i,:])}")
            random_rate += np.average(self.random_data_rate_log[i,:])
            print(f"{i}-th Agent's episode average data rate (CG based): {np.average(self.channel_based_data_rate_log[i,:])}")
            CG_based_rate += np.average(self.channel_based_data_rate_log[i,:])
            print(f"{i}-th Agent's episode average data rate (MAX_C_based_rate based): {np.average(self.max_available_channel_based_data_rate_log[i,:])}")
            MAX_C_based_rate += np.average(self.max_available_channel_based_data_rate_log[i,:])
        print(f"MADL MAX data rate: {np.max(self.rate_data_log)}")
        print(f"MADL episode average data rate:{np.average(self.rate_data_log)}")
        print(f"MVT episode average data rate:{np.average(self.MVT_data_rate_log)}")
        print(f"RANDOM episode average data rate:{random_rate/self.GS_size}")
        print(f"CG-based episode MAX data rate:{np.max(self.channel_based_data_rate_log)}")
        print(f"CG-based episode average data rate:{np.average(self.channel_based_data_rate_log)}")
        print(f"MAX_C_based_rate-based episode MAX data rate:{np.max(self.max_available_channel_based_data_rate_log)}")
        print(f"MAX_C_based_rate-based episode average data rate:{MAX_C_based_rate/self.GS_size}")

        # average througput
        plt.figure(10)
        interval = 15 # mark interveal
        MADRL_data_rate = self.rate_data_log.mean(axis=0)
        moving_MADRL_data_rate = np.convolve(MADRL_data_rate, np.ones(interval), 'valid')/interval
        MVT_data_rate = self.MVT_data_rate_log.mean(axis=0)
        moving_MVT_data_rate = np.convolve(MVT_data_rate, np.ones(interval), 'valid')/interval
        CG_data_rate = self.channel_based_data_rate_log.mean(axis=0)
        moving_CG_data_rate = np.convolve(CG_data_rate, np.ones(interval), 'valid')/interval
        Ranmdom_data_rate = self.random_data_rate_log.mean(axis=0)
        moving_Ranmdom_data_rat = np.convolve(Ranmdom_data_rate, np.ones(interval), 'valid')/interval

        time_step = np.arange(self.terminal_time-interval+1); 
        plt.plot(time_step, moving_MADRL_data_rate[1:], label='MADQN', marker='*', markevery=interval)
        plt.plot(time_step, moving_MVT_data_rate[1:], label='MVT', marker='.', markevery=interval)        
        plt.plot(time_step, moving_CG_data_rate[1:], label='MAX-CG', marker='P', markevery=interval) 
        plt.plot(time_step, moving_Ranmdom_data_rat[1:], label='Random', marker='|', markevery=interval)
        plt.xlim((1,self.terminal_time-interval+1))
        plt.ylabel("Moving average throughput [bps]"); plt.xlabel('time step'); plt.grid(); plt.legend()

        # Plot Agents' Status
        plt.figure(1)
        agents = np.arange(0,self.GS_size)
        _status = np.zeros((self.GS_size, 5))
        for i in range(self.GS_size):
            _status[i][0] = np.count_nonzero(self.agent_status_log[i] == 1)
            _status[i][1] = np.count_nonzero(self.agent_status_log[i] == 2)
            _status[i][2] = np.count_nonzero(self.agent_status_log[i] == 3)
            _status[i][3] = np.count_nonzero(self.agent_status_log[i] == 4)

        bar_width = 0.1
        status_1 = plt.bar(agents, _status[:,0], bar_width, label='HOF-non_service')
        status_2 = plt.bar(agents + bar_width, _status[:,1], bar_width, label='HOF-QoS')
        status_3 = plt.bar(agents + 2*bar_width, _status[:,2], bar_width, label='HOC')
        status_4 = plt.bar(agents + 3*bar_width, _status[:,3], bar_width, label='ACK')
        plt.xticks(np.arange(bar_width, self.GS_size+bar_width,1), agents)
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
        status_3 = plt.bar(agents + 2*bar_width, MVT_status[:,2], bar_width, label='HOC')
        status_4 = plt.bar(agents + 3*bar_width, MVT_status[:,3], bar_width, label='ACK')
        plt.xticks(np.arange(bar_width, self.GS_size+bar_width,1), agents)
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
        status_3 = plt.bar(agents + 2*bar_width, Random_status[:,2], bar_width, label='HOC')
        status_4 = plt.bar(agents + 3*bar_width, Random_status[:,3], bar_width, label='ACK')
        plt.xticks(np.arange(bar_width, self.GS_size+bar_width,1), agents)
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
        status_3 = plt.bar(agents + 2*bar_width, channel_based_status[:,2], bar_width, label='HOC')
        status_4 = plt.bar(agents + 3*bar_width, channel_based_status[:,3], bar_width, label='ACK')        
        plt.xticks(np.arange(bar_width, self.GS_size+bar_width,1), agents)
        plt.xlabel('# of Agent'); plt.legend(); plt.title("Channel-gain-based")

        ## UEs' data rate
        plt.figure(0)
        UEs_data_rate = np.zeros((self.GS_size, 4))
        UEs_data_rate[:,0] = self.rate_data_log.mean(axis=1)
        UEs_data_rate[:,1] = self.MVT_data_rate_log.mean(axis=1)
        UEs_data_rate[:,2] = self.channel_based_data_rate_log.mean(axis=1)
        UEs_data_rate[:,3] = self.random_data_rate_log.mean(axis=1)
        bar_width = 0.1
        status_1 = plt.bar(agents, UEs_data_rate[:,0], bar_width, label='MADQN')
        status_2 = plt.bar(agents + bar_width, UEs_data_rate[:,1], bar_width, label='MVT')
        status_3 = plt.bar(agents + 2*bar_width, UEs_data_rate[:,2], bar_width, label='MAX-CG')
        status_5 = plt.bar(agents + 3*bar_width, UEs_data_rate[:,3], bar_width, label='Random')
        plt.xticks(np.arange(bar_width, self.GS_size+bar_width,1), agents)
        plt.ylabel("Moving average throughput [bps]"); plt.xlabel('UEs'); plt.legend()

        # ACK variance
        print(f"ACK var --> MADQN: {np.var(_status[:,3])}, MVT: {np.var(MVT_status[:,3])}, Random: {np.var(Random_status[:,3])}, SINR: {channel_based_status[:,3]}")
        
        # Plot comparsion of Ack HO strategies
        plt.figure(7)
        bar_width = 0.1
        status_1 = plt.bar(agents, _status[:,3], bar_width, label='MADQN')
        status_2 = plt.bar(agents + bar_width, MVT_status[:,3], bar_width, label='MVT')
        status_3 = plt.bar(agents + 2*bar_width, channel_based_status[:,3], bar_width, label='MAX-CG')
        status_5 = plt.bar(agents + 3*bar_width, Random_status[:,3], bar_width, label='Random')
        plt.xticks(np.arange(bar_width, self.GS_size+bar_width,1), agents)
        plt.xlabel('UEs'); plt.legend(); plt.ylabel('Communication times')

        # Plot comparsion of average handover
        plt.figure(5)
        HO_MADQN = np.zeros((self.terminal_time+1))
        HO_MVT   = np.zeros((self.terminal_time+1))
        HO_RANDOM   = np.zeros((self.terminal_time+1))
        HO_CG = np.zeros((self.terminal_time+1))
        HO_MAX_C = np.zeros((self.terminal_time+1))
        for agent in range(self.GS_size):
            for t in range(self.terminal_time+1):
                if self.HO_log[agent][t]:
                    HO_MADQN[t:] += 1
                if self.MVT_HO_log[agent][t]:
                    HO_MVT[t:] += 1
                if self.ramdon_HO_log[agent][t]:
                    HO_RANDOM[t:] += 1
                if self.CG_HO_log[agent][t]:
                    HO_CG[t:] += 1
                if self.MAC_HO_log[agent][t]:
                    HO_MAX_C[t:] += 1
        
        HO_MADQN[:] /= self.GS_size
        HO_MVT[:] /= self.GS_size
        HO_RANDOM[:] /= self.GS_size
        HO_CG[:] /= self.GS_size
        HO_MAX_C[:] /= self.GS_size

        HO_MADQN[:] = HO_MADQN[-1] / self.terminal_time
        HO_MVT[:] = HO_MVT[-1] / self.terminal_time
        HO_CG[:] = HO_CG[-1] / self.terminal_time
        HO_RANDOM[:] = HO_RANDOM[-1] / self.terminal_time
        HO_MAX_C[:] = HO_MAX_C[-1] / self.terminal_time
        interval = 10 # mark interveal
        time_step = np.arange(self.terminal_time)

        plt.plot(time_step, HO_MADQN[1:], label='MADQN', marker='*', markevery=interval)

        plt.plot(time_step, HO_MVT[1:], label='MVT', marker='.', markevery=interval)        

        plt.plot(time_step, HO_CG[1:], label='MAX-CG', marker='P', markevery=interval)

        #plt.plot(time_step, HO_MAX_C[1:], label='MAC', marker='2', markevery=interval)

        plt.plot(time_step, HO_RANDOM[1:], label='Random', marker='|', markevery=interval)

        plt.xlim((1,155))
        plt.ylabel('Average handover'); plt.legend(); plt.xlabel('time step'); plt.grid()


        # Plot comprasion of communication failure rate
        plt.figure(6)
        HOF_MADQN = np.zeros((self.terminal_time+1))
        HOF_MVT   = np.zeros((self.terminal_time+1))
        HOF_RANDOM   = np.zeros((self.terminal_time+1))
        HOF_CG  = np.zeros((self.terminal_time+1))
        HOF_MAX_C  = np.zeros((self.terminal_time+1))
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
                if self.max_available_channel_based_status_log[agent][t] == 1 or self.max_available_channel_based_status_log[agent][t] == 2:
                    HOF_MAX_C[t:] += 1

        HOF_MADQN /= self.GS_size
        HOF_MVT   /= self.GS_size
        HOF_RANDOM   /= self.GS_size
        HOF_CG  /= self.GS_size
        HOF_MAX_C  /= self.GS_size

        _HOF_MADQN = (HOF_MADQN[-1]) / self.terminal_time
        _HOF_MVT   = (HOF_MVT[-1]) / self.terminal_time
        _HOF_RANDOM   = (HOF_RANDOM[-1]) / self.terminal_time
        _HOF_CG  = (HOF_CG[-1]) / self.terminal_time
        _HOF_MAX_C  = (HOF_MAX_C[-1]) / self.terminal_time
        
        HOF_MADQN[:] = _HOF_MADQN
        HOF_MVT[:]   = _HOF_MVT  
        HOF_RANDOM[:]   = _HOF_RANDOM  
        HOF_CG[:]  = _HOF_CG 
        HOF_MAX_C[:]  = _HOF_MAX_C 

        time_step = np.arange(self.terminal_time)
        interval = 15 # mark interveal

        plt.plot(time_step, HOF_MADQN[1:], label='MADQN', marker='*', markevery=interval)

        plt.plot(time_step, HOF_MVT[1:], label='MVT', marker='.', markevery=interval)        

        plt.plot(time_step, HOF_CG[1:], label='MAX-CG', marker='P', markevery=interval)

        #plt.plot(time_step, HOF_MAX_C[1:], label='MAC', marker='2', markevery=interval)

        plt.plot(time_step, HOF_RANDOM[1:], label='Random', marker='|', markevery=interval)

        plt.xlim((1,155))
        plt.ylabel("Average communication failure rate"); plt.legend(loc=(0.02, 0.5)); plt.xlabel('time step'); plt.grid()

        # Data rate log
        # plt.figure(8)
        # time_step = np.arange(self.terminal_time+1)
        # for i in range(self.GS_size):
        #     plt.step(time_step, self.rate_data_log[i])
        # plt.title("Agents' Data rate log")

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


