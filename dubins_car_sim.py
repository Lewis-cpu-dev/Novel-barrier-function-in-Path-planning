import numpy as np
import shapely
from typing import List,Optional,Tuple
import shapely
from scipy.spatial import ConvexHull
import math
import casadi
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate



def in_bound(bound : tuple, s):
    return (bound[0] <= s) and (s <= bound[1]) 

class DubinsCarStateSpace:
    """Just the dynamics information of a 2nd order Dubins car.
    
    State is a 5-list, control is a 2-list.
    """
    def __init__(self, t_step : float, car_L : float, car_w : float, a_bound : tuple, psi_bound : tuple, v_bound : tuple, phi_bound : tuple):
        """
        Initialize the Dubins car with the given parameters.
        """
        self.t_step = t_step
        self.car_L = car_L
        self.car_w = car_w
        self.a_bound = a_bound
        self.psi_bound = psi_bound

        self.v_bound = v_bound
        self.phi_bound = phi_bound
    
    def state_to_dict(self, state : list) -> dict:
        """
        Convert a state to a dict.
        Input:
            state: list of 5 floats, [x, y, theta, v, phi]
        Output:
            dict: dictionary
        """
        x, y, theta, v, phi = state
        return {'x':x, 'y':y, 'theta':theta, 'v':v, 'phi':phi}

    def control_to_dict(self, control : list) -> dict:
        """
        Convert a control to a dict.
        Input:
            control: list of 2 floats, [a, psi]
        Output:
            dict: dictionary
        """
        a, psi = control
        return {'a':a, 'psi':psi}

    def zero_state(self) -> list:
        return [0.0]*5

    def zero_control(self) -> list:
        return [0.0]*2

    def state_in_bound(self, state : list) -> bool:
        """
        Check if the velocity and steering angle are within bounds.
        Input:
            state: list of 5 floats, [x, y, theta, v, phi]
        Output:
            bool_bound: boolean, True if in bounds, False otherwise
        """
        x, y, theta, v, phi = state
        v_in_bound = in_bound(self.v_bound, v)
        phi_in_bound = in_bound(self.phi_bound, phi)
        return v_in_bound and phi_in_bound

    def control_in_bound(self, control : list) -> bool:
        """
        Check if the velocity and steering angle are within bounds.
        Output:
            bool_bound: boolean, True if in bounds, False otherwise
        """
        a, psi = control
        a_in_bound = in_bound(self.a_bound, a)
        psi_in_bound = in_bound(self.psi_bound, psi)
        return a_in_bound and psi_in_bound

    def sample_state(self, x_bound, y_bound) -> list:
        """
        Sample a random state for the car.
        Input:
            x_bound: list of 2 floats, [x_min, x_max]
            y_bound: list of 2 floats, [y_min, y_max]
        Output:
            q_lst: list of 5 floats, [x, y, theta, v, phi]
        """
        x_rand = np.random.uniform(*x_bound)
        y_rand = np.random.uniform(*y_bound)
        theta_rand = np.random.uniform(0, 2*np.pi)
        v_rand = np.random.uniform(*self.v_bound)
        phi_rand = np.random.uniform(*self.phi_bound)
        return [ x_rand, y_rand, theta_rand, v_rand, phi_rand ]

    def sample_control(self) -> list:
        """
        Sample a random control for the car.
        Output:
            u_lst: list of 2 floats, [a, psi]
        """
        a_rand = np.random.uniform(*self.a_bound)
        psi_rand = np.random.uniform(*self.psi_bound)
        return [ a_rand, psi_rand ]
       
    def dynamics(self, state: list, control: list) -> list:
        """Returns the derivative of the state with respect to time
        given the control.
        
        Input:
            state: list of 5 floats, [x, y, theta, v, phi]
            control: list of 2 floats, [a, psi]
        Output:
            dstate: list of 5 floats, [dx/dt, dy/dt, dtheta/dt, dv/dt, dphi/dt]
        """
        x,y,theta,v,phi = state
        a,psi = control
        return [v*np.cos(theta),v*np.sin(theta),(v/self.car_L)*np.tan(phi),a,psi]
                
    def step(self, state : list, control : list, t_step=None) -> list:
        """
            Simple Euler integration step for the car.
            Input:
                state: list of 5 floats, [x, y, theta, v, phi]
                control: list of 2 floats, [a, psi]
                t_step: float, or None (in which case self.t_step is used)
            Output:
                new_state: list of 5 floats, [x, y, theta, v, phi]
        """
        if t_step is None:
            t_step = self.t_step
        x,y,theta,v,phi = state
        a,psi = control
        a = np.clip(a, *self.a_bound)
        psi = np.clip(psi, *self.psi_bound)
        assert in_bound(self.a_bound, a)
        assert in_bound(self.psi_bound, psi)
        ds = self.dynamics(state, control)
        return [v+dv*t_step for v,dv in zip(state,ds)]

    def integrate(self, state : list, control : list, T_total : float) -> list:
        """
        Integrate the car for a given time horizon with a constant control
        using Euler integration.
        Input:
            state: list of 5 floats, [x, y, theta, v, phi]
            control: list of 2 floats, [a, psi]
            T_total: float, total time to integrate
        Output:
            state_lst: list of states, [[x, y, theta, v, phi], ...]
        """
        state_lst = [state]
        t = 0
        while t+self.t_step < T_total:
            state = self.step(state, control)
            state_lst.append(state)
            t += self.t_step
        if t < T_total:
            state_lst.append(self.step(state, control, T_total - t))
        return state_lst

    def rollout(self, state : list, policy : callable, T_total : float, t_start:float=0.0) -> Tuple[list,list,list]:
        """
        Integrate the car for a given time horizon with a policy
        using Euler integration.
        Input:
            state: list of 5 floats, [x, y, theta, v, phi]
            policy: function u(x,t) -> [a, psi]
            T_total: float, total time to integrate
        Output:
            (t_lst,state_lst,u_list): list of times, states, and controls
        """
        t_list = [t_start]
        state_list = [state]
        u_list = []
        t = 0
        while t+self.t_step < T_total:
            control = policy(state, t_start+t)
            state = self.step(state, control)
            t_list.append(t_start+t+self.t_step)
            state_list.append(state)
            u_list.append(control)
            t += self.t_step
        if t < T_total:
            control = policy(state, t_start+t)
            t_list.append(t_start+T_total)
            state_list.append(self.step(state, control, T_total - t))
            u_list.append(control)
        return t_list,state_list,u_list
    
    def rollout_drag(self, state : list, policy : callable, T_total : float, t_start:float=0.0) -> Tuple[list,list,list]:
        """
        Integrate the car for a given time horizon with a policy
        using Euler integration.
        Input:
            state: list of 5 floats, [x, y, theta, v, phi]
            policy: function u(x,t) -> [a, psi]
            T_total: float, total time to integrate
        Output:
            (t_lst,state_lst,u_list): list of times, states, and controls
        """
        t_list = [t_start]
        state_list = [state]
        u_list = []
        t = 0
        while t+self.t_step < T_total:
            control = policy(state, t_start+t)
            state = self.step(state, control)
            t_list.append(t_start+t+self.t_step)
            state_list.append(state)
            u_list.append(control)
            t += self.t_step
        if t < T_total:
            control = policy(state, t_start+t)
            t_list.append(t_start+T_total)
            state_list.append(self.step(state, control, T_total - t))
            u_list.append(control)
        return t_list,state_list,u_list
    def rollout_PD(self, state : list, control : list, policy : callable, T_total : float, t_start:float=0.0) -> Tuple[list,list,list]:
        """
        Integrate the car for a given time horizon with a policy
        using Euler integration.
        Input:
            state: list of 5 floats, [x, y, theta, v, phi]
            policy: function u(x,t) -> [a, psi]
            T_total: float, total time to integrate
        Output:
            (t_lst,state_lst,u_list): list of times, states, and controls
        """
        t_list = [t_start]
        state_list = [state]
        u_list = [control]
        t = 0
        while t+self.t_step < T_total:
            control = policy(state,control,t_start+t)
            state = self.step(state, control)
            t_list.append(t_start+t+self.t_step)
            state_list.append(state)
            u_list.append(control)
            t += self.t_step
        if t < T_total:
            control = policy(state,control, t_start+t)
            t_list.append(t_start+T_total)
            state_list.append(self.step(state, control, T_total - t))
            # u_list.append(control)
        return t_list,state_list,u_list
    
    def rollout_noise(self, state : list, control : list, policy : callable, T_total : float, t_start:float=0.0) -> Tuple[list,list,list]:
            """
            Integrate the car for a given time horizon with a policy
            using Euler integration.
            Input:
                state: list of 5 floats, [x, y, theta, v, phi]
                policy: function u(x,t) -> [a, psi]
                T_total: float, total time to integrate
            Output:
                (t_lst,state_lst,u_list): list of times, states, and controls
            """
            t_list = [t_start]
            state_list = [state]
            u_list = []
            t = 0
            while t+self.t_step < T_total:
                sigma_a,sigma_psi = 1.5,0.7
                noise = [np.random.normal(0, sigma_a), np.random.normal(0, sigma_psi)]
                control = [a+b for a,b in zip(policy(state, control,t_start+t),noise)]
                state = self.step(state, control)
                t_list.append(t_start+t+self.t_step)
                state_list.append(state)
                u_list.append(control)
                t += self.t_step
            if t < T_total:
                sigma_a,sigma_psi = 1.5,0.7
                noise = [np.random.normal(0, sigma_a), np.random.normal(0, sigma_psi)]
                control = [a+b for a,b in zip(policy(state, control,t_start+t),noise)]
                t_list.append(t_start+T_total)
                state_list.append(self.step(state, control, T_total - t))
                u_list.append(control)
            return t_list,state_list,u_list


class DubinsCarCSpace:
    """A container for all the dynamics, sampling, and collision checking
    information for a Dubins car with obstacles."""
    def __init__(self, dynamics : DubinsCarStateSpace,
                 x_bound : tuple,
                 y_bound : tuple,
                 car_reference_shape : shapely.geometry.Polygon,
                 obstacles : List[shapely.geometry.Polygon] = [],
                 obstacle_points: List = [],
                 obstacle_radius=0,
                 distance_weights : Optional[list] = None):
        self.dynamics = dynamics
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.car_reference_shape = car_reference_shape
        self.obstacles = obstacles
        self.obstacles_points = obstacle_points
        self.obstacle_radius = obstacle_radius
        if distance_weights is None:
            self.distance_weights = [0]*len(dynamics.zero_state())
        else:
            self.distance_weights = distance_weights

    def sample_state(self) -> list:
        """Samples a random state."""
        return self.dynamics.sample_state(self.x_bound, self.y_bound)

    def state_valid(self, state : list) -> bool:
        """State feasibility checking."""
        if not self.dynamics.state_in_bound(state):
            return False
        if not in_bound(self.x_bound,state[0]):
            return False
        if not in_bound(self.y_bound,state[1]):
            return False
        #TODO: collision checking with obstacles
        #SOLUTION 3.B
        car_polygon = self.car_shape(state)
        if car_polygon is None:
            return True
        for obstacle in self.obstacles:
            if car_polygon.intersects(obstacle):
                return False
        return True
    
    def inevitable_collision(self, state : list) -> bool:
        #SOLUTION 3.C
        x,y,theta,v,phi = state
        if v >= 0:
            max_distance_traveled = -0.5 * v**2 / self.dynamics.a_bound[0]
        else:
            max_distance_traveled = -0.5 * v**2 / self.dynamics.a_bound[1]
        x2 = x + max_distance_traveled*np.cos(theta)
        y2 = y + max_distance_traveled*np.sin(theta)
        if not in_bound(self.x_bound,x2):
            return True
        if not in_bound(self.y_bound,y2):
            return True
        car_polygon = self.car_shape([x2,y2,theta,v,phi])
        for obstacle in self.obstacles:
            if car_polygon.intersects(obstacle):
                return True
        return False

    def traj_valid(self, traj : List[list]) -> bool:
        """Trajectory feasibility checking."""
        for state in traj:
            if not self.state_valid(state):
                return False
        return True

    def distance(self, state1 : list, state2 : list) -> float:
        #TODO: there's a problem here
        diff = np.array(state1) - np.array(state2)
        #SOLUTION
        diff[2] = np.mod(diff[2] + np.pi, 2*np.pi) - np.pi
        diff = np.multiply(diff,self.distance_weights)
        return np.sqrt(np.dot(diff,diff))

    def interpolate(self, state1 : list, state2 : list, u : float) -> float:
        #TODO: there's a problem here
        #SOLUTION
        x = [ a+u*(b-a) for a,b in zip(state1,state2) ]
        x[2] = state1[2] + u*(np.mod(x[2] - state1[2] + np.pi, 2*np.pi) - np.pi)
        return x
        return [ a+u*(b-a) for a,b in zip(state1,state2) ]
    
    def car_shape(self, state : list) -> shapely.geometry.Polygon:
        x, y, theta, v, phi = state
        #SOLUTION
        if self.car_reference_shape is None:
            return None
        x,y,theta,v,phi = state
        R = np.array([[np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]])
        car_polygon = shapely.transform(self.car_reference_shape, lambda pts: np.dot(R, pts.T).T + np.array([x, y]))
        return car_polygon
        #TODO: implement this
        return self.car_reference_shape
    
    def sample_random_obstacle(self):
        total = []  
        all_points = []
        for _ in range(3):
            while True:  
                num = math.floor(30 * np.random.uniform(0.4, 1))  
                points = (np.random.rand(num, 2) * 2 - (np.random.uniform(-6,6),np.random.uniform(-4,4))) * 2
                hull = ConvexHull(points)
                convex_points = points[hull.vertices]
                print('1111:',convex_points)
                obstacle = shapely.geometry.Polygon(convex_points)

                overlap = False
                for existing_obstacle in total:

                    if obstacle.intersects(existing_obstacle):
                        overlap = True
                        break
                if not overlap:
                    if obstacle.intersects(shapely.geometry.Polygon([[1,0],[0,1],[-1,0],[0,-1]])) \
                        or obstacle.intersects(shapely.geometry.Polygon([[0,6],[-1,5],[0,4],[1,5]])):
                        continue
                    total.append(obstacle)
                    all_points.append(list(convex_points))
                    break
        self.obstacles = total
        self.obstacles_points = all_points
        print(f'\n\n convex {all_points}\n\n')
        return enumerate(total), all_points
    
    def sample_fail_random_obstacle(self):

        total = []  
        all_points = []
        points= [[1,2],[1,3],[-1,3],[-1,2]]
        all_points.append(points)
        total.append(shapely.geometry.Polygon(points))
        for _ in range(3):
            while True:  
                num = math.floor(30 * np.random.uniform(1, 4))  
                points = (np.random.rand(num, 2) * 2 - (np.random.uniform(-2,2),np.random.uniform(-3,3))) * 2
                hull = ConvexHull(points)
                convex_points = points[hull.vertices]
                print('1111:',convex_points)
                obstacle = shapely.geometry.Polygon(convex_points)

                overlap = False
                for existing_obstacle in total:

                    if obstacle.intersects(existing_obstacle):
                        overlap = True
                        break
                if not overlap:
                    if obstacle.intersects(shapely.geometry.Polygon([[1,0],[0,1],[-1,0],[0,-1]])) \
                        or obstacle.intersects(shapely.geometry.Polygon([[0,6],[-1,5],[0,4],[1,5]])):
                        continue
                    total.append(obstacle)
                    all_points.append(list(convex_points))
                    break
        self.obstacles = total
        self.obstacles_points = all_points
        print(f'\n\n convex {all_points}\n\n')
        return enumerate(total), all_points
    def create_vehicle_polygon(self,x, y, theta):
        # Vehicle polygon with rear wheel center at (x, y) and oriented by theta
        vehicle_shape = Polygon([[-self.dynamics.car_L/2, -self.dynamics.car_w/2], 
                                 [-self.dynamics.car_L/2, self.dynamics.car_w/2], 
                                 [self.dynamics.car_L/2, self.dynamics.car_w/2], 
                                 [self.dynamics.car_L/2, -self.dynamics.car_w/2]])
        # Rotate and translate the vehicle shape to the current position and orientation
        rotated_vehicle = rotate(vehicle_shape, theta, origin=(0, 0), use_radians=True)
        translated_vehicle = translate(rotated_vehicle, xoff=x, yoff=y)
        return translated_vehicle
    

    # Check for collision between vehicle and obstacle
    def check_collision(self,vehicle, obstacle):
        return vehicle.intersects(obstacle)








