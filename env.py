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

"""
import matplotlib.pyplot as plt
from copy import copy

import numpy as np
from gymnasium import spaces

from pettingzoo.utils import agent_selector
from pettingzoo import AECEnv

class LEOSATEnv(AECEnv):
    def __init__(self, render_mode=None, debugging=False) -> None:        
        # Agent area
        self.GS_area_max_x = 100    # km
        self.GS_area_max_y = 100    # km
        self.GS_area_max_z = 1_000  # km

        self.GS_size = 10
        self.GS = np.zeros((self.GS_size, 3)) # coordinate (x, y, z) of GS

        self.timestep = None
        self.terminal_time = 155 # s

        self.SAT_len = 22
        self.SAT_plane = 2 # of plane
        self.SAT_coverage_radius = 55 # km
        self.SAT_speed = 7.9 # km/s, caution!! direction
        self.theta = np.linspace(0, 2 * np.pi, 150)
        self.SAT_point = np.zeros((self.SAT_len * self.SAT_plane, 3)) # coordinate (x, y, z) of SAT center point
        self.SAT_coverage = np.zeros((self.SAT_len * self.SAT_plane, 3, 150)) # coordinate (x, y, z) of SAT coverage
        self.SAT_point[:,2] = 500 # km, SAT height 
        self.SAT_coverage[:,2,:] = 500 # km, SAT height
        self.SAT_Load = np.full(self.SAT_len*self.SAT_plane, 5) # the available channels of SAT
        self.SAT_W = 10 # MHz BW budget of SAT

        self.service_indicator = np.zeros((self.GS_size, self.SAT_len*self.SAT_plane)) # indicator: users are served by SAT (one-hot vector)

        self.agents = [f"groud_station_{i}" for i in range(self.GS_size)]
        self.possible_agents = self.agents[:]

        self._none = self.SAT_len * self.SAT_plane
        self.action_spaces = {i: spaces.Discrete(self.SAT_len * self.SAT_plane) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(3, self.SAT_len * self.SAT_plane), dtype=np.int8
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


    def _SAT_coordinate(self, SAT, SAT_len, time, speed):
        """
        return real-time SAT center point
        """
        _SAT = np.copy(SAT)
        for i in range(SAT_len):
            _SAT[i,0] = 65*i -speed * time
            _SAT[i,1] = 10

            _SAT[i + SAT_len,0] = -25 + 65*i -speed * time
            _SAT[i+ SAT_len,1] =  10 + 65
        
        return _SAT

    def _SAT_coverage_position(self, SAT_coverage, SAT_len, time, speed, radius, theta):
        """
        return real-time SAT coverage position
        for render
        """
        _SAT_coverage = np.copy(SAT_coverage)
        for i in range(SAT_len):
            _SAT_coverage[i,0,:] = 65*i -speed * time + radius * np.cos(theta)
            _SAT_coverage[i,1,:] =  10                + radius * np.sin(theta)

            _SAT_coverage[i + SAT_len,0,:] = -25 + 65*i -speed * time + radius * np.cos(theta)
            _SAT_coverage[i + SAT_len,1,:] =  10 +   65               + radius * np.sin(theta)

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

    def _get_visible_time(self, SAT_point, SAT_speed, coverage_radius, GS):
        """
        return visible time btw SAT and GS
        """
        _num = np.max((coverage_radius ** 2 - (GS[1]-SAT_point[1]) ** 2, 0))
        visible_time = (np.sqrt(_num) - GS[0] + SAT_point[0]) / SAT_speed
        visible_time = np.max((visible_time, 0))
        
        return visible_time        


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

        # observations
        self.observations = {}
        for i in range(self.GS_size):
            observation = (
                self.coverage_indicator[i],
                self.SAT_Load,
                self.visible_time[i]
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
            # rewards
            for i in range(self.GS_size):
                reward = 0

                # non-coverage area
                if self.coverage_indicator[i][self.states[self.agents[i]]] == 0:
                    reward = -20
                # HO occur
                elif self.service_indicator[i][self.states[self.agents[i]]] == 0:
                    reward = -10
                else:
                # Overload
                    _actions = np.array(list(self.states.values()))
                    if np.count_nonzero(_actions == _actions[i]) > self.SAT_Load[_actions[i]]:
                        reward = -5
                    else:
                        reward = self.visible_time[i][_actions[i]]
                self.rewards[self.agents[i]] = reward
        
            # Update service indicator
            self.service_indicator = np.zeros((self.GS_size, self.SAT_len*self.SAT_plane))
            for i in range(self.GS_size):
                self.service_indicator[i][self.states[self.agents[i]]] = 1
    
            # Check termination conditions
            if self.timestep == self.terminal_time:
                self.terminations = {agent: True for agent in self.agents}

            # Get obersvations
            self.timestep += 1

            # Get SAT position
            self.SAT_point = self._SAT_coordinate(self.SAT_point, self.SAT_len, self.timestep, self.SAT_speed)
            # Get coverage indicator
            self.coverage_indicator = self._is_in_coverage(self.SAT_point, self.GS, self.SAT_coverage_radius)
            # Get visible time        
            for i in range(self.GS_size):
                for j in range(self.SAT_len*2):
                    self.visible_time[i][j] = self._get_visible_time(self.SAT_point[j], self.SAT_speed, self.SAT_coverage_radius, self.GS[i])
            
            for i in range(self.GS_size):
                observation = (
                    self.coverage_indicator[i],
                    self.SAT_Load,
                    self.visible_time[i]
                )
                self.observations[f"groud_station_{i}"] = observation
            
            if self.render_mode == "human":
                self.render()
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
        
        SAT_area = self._SAT_coverage_position(self.SAT_coverage, self.SAT_len, self.timestep, self.SAT_speed, self.SAT_coverage_radius, self.theta)        

        # Plot SAT's coverage area
        for i in range(self.SAT_len * self.SAT_plane):
            axes.plot(SAT_area[i,0,:], SAT_area[i,1,:])
        # Plot SAT's point
        axes.plot(self.SAT_point[:,0], self.SAT_point[:,1], 'o')
        
        # Plot ground station's area
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
        
        axes.set_aspect(1)
        axes.axis([-50, 200, -50, 150])

        plt.show()