from dubins_car_sim import DubinsCarStateSpace,DubinsCarCSpace
import shapely
import shapely.plotting
from shapely.geometry import Polygon,LineString
import numpy as np
import casadi
import casadi as ca
import time
from typing import Union, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import os,sys


def generate_polygon(centers, radius, num_sides=20):
    """
    Generates the vertices of a regular polygon approximating a circle.

    Parameters:
    - center: Tuple of (c_x, c_y)
    - radius: Radius of the circle
    - num_sides: Number of sides of the polygon

    Returns:
    - vertices: A list of (x, y) tuples representing the polygon's vertices
    """
    obstacles = []
    vertices_obs=[]
    for center in centers:
        c_x, c_y = center
        angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
        vertices = [(c_x + radius * np.cos(angle), c_y + radius * np.sin(angle)) for angle in angles]
        polygon = Polygon(vertices)
        obstacles.append(polygon)
        vertices_obs.append(vertices)
    return obstacles, vertices_obs

def setup_system():
    t_step, car_L = 0.1, 0.5

    t_bound = [0.0, 1.0]
    x_bound = [-10.0, 10.0]
    y_bound = [-10.0, 10.0]

    a_bound = [-10.0, 10.0]
    psi_bound = [-3.0, 3.0]

    v_bound = [-5.0, 5.0]
    phi_bound = [-np.pi/3, np.pi/3]
    dynamics = DubinsCarStateSpace(t_step, car_L, a_bound, psi_bound, 
                    v_bound, phi_bound)

    car_w, car_h = 0.75, 0.3

    dist_weights = np.array([1.0, 1.0, 1.0, 0.2, 0.1])
    car_shape = None

    car_points = np.array([[-car_w*0.1, -car_h/2], [ car_w*0.9, -car_h/2], 
                            [ car_w*0.9,  car_h/2], [-car_w*0.1,  car_h/2]])
    car_shape = shapely.geometry.Polygon(car_points)
    obstacles = []
    obstacles_Polygon=[]
    # for obstacle in obstacles:
    #     obstacles_Polygon.append(shapely.geometry.Polygon(obstacle))
    
    #TODO: you may want to set up obstacles here
    dubins = DubinsCarCSpace(dynamics, x_bound, y_bound, car_shape, obstacles_Polygon,obstacles, dist_weights)
    # _, _ = dubins.sample_random_obstacle()
    return dubins

def setup_system_circular():
    t_step, car_L,car_w = 0.1, 0.5,0.2

    t_bound = [0.0, 1.0]
    x_bound = [-10.0, 10.0]
    y_bound = [-10.0, 10.0]

    a_bound = [-10.0, 10.0]
    psi_bound = [-3.0, 3.0]

    v_bound = [-5.0, 5.0]
    phi_bound = [-np.pi/3, np.pi/3]
    dynamics = DubinsCarStateSpace(t_step, car_L, car_w, a_bound, psi_bound, 
                    v_bound, phi_bound)

    car_w, car_h = 0.75, 0.3

    dist_weights = np.array([1.0, 1.0, 1.0, 0.2, 0.1])
    car_shape = None

    car_points = np.array([[-car_w*0.1, -car_h/2], [ car_w*0.9, -car_h/2], 
                            [ car_w*0.9,  car_h/2], [-car_w*0.1,  car_h/2]])
    car_shape = shapely.geometry.Polygon(car_points)
    d1=0.8
    d2 = 0.6

    centres_ = [
                [d1,[0,2]]
                ,[d1,[0,2],[-4,3],[0,4]]
                ,[d1,[0,2],[-4,3],[0,4],[-2,5],[4,0]]
                ,[d1,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2]]
               ,[d1,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2],[-3,5],[2,0]]
               ,[d1,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2],[-3,5],[2,0],[1,10],[4,10],[3,11],[-2,13],[-1,10],[5,13]]
               ,[d2,[0,2]]
                ,[d2,[0,2],[-4,3],[0,4]]
                ,[d2,[0,2],[-4,3],[0,4],[-2,5],[4,0]]
                ,[d2,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2]]
               ,[d2,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2],[-3,5],[2,0]]
                ,[d2,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2],[-3,5],[2,0],[1,10],[4,10],[3,11],[-2,13],[-1,10],[5,13]]]


    centres_ = [
                [d1,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2],[-3,5],[2,0]]
               ,[d1,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2],[-3,5],[2,0],[1,10],[4,10],[3,11],[-2,13],[-1,10],[5,13]]
            ,[d2,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2],[-3,5],[2,0]],
            [d2,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2],[-3,5],[2,0],[1,10],[4,10],[3,11],[-2,13],[-1,10],[5,13]]]
    # centres_ = [[d1,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2]]]   
    centres_ = [
                [d1,[0,2]]
                ,[d1,[0,2],[-4,3],[0,4]]
                ,[d1,[0,2],[-4,3],[0,4],[-2,5],[4,0]]
                ,[d1,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2]]
               ,[d1,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2],[-3,5],[2,0]]
               ,[d2,[0,2]]
                ,[d2,[0,2],[-4,3],[0,4]]
                ,[d2,[0,2],[-4,3],[0,4],[-2,5],[4,0]]
                ,[d2,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2]]
               ,[d2,[0,2],[-4,3],[0,4],[-2,5],[4,0],[4,2],[1.5,2],[-3,5],[2,0]]]

    centres = np.array(centres_[2][1:] ,dtype=float)
    # centres = []
    r_1=0.8
    # for obstacle in obstacles:
    #     obstacles_Polygon.append(shapely.geometry.Polygon(obstacle))
    obstacles_Polygon,obstacles=generate_polygon(centres, r_1)
    #TODO: you may want to set up obstacles here
    dubins = DubinsCarCSpace(dynamics, x_bound, y_bound, car_shape, obstacles_Polygon,centres,r_1, dist_weights)
    # _, _ = dubins.sample_random_obstacle()
    return dubins, centres_

class PerturbedDubinsCarStateSpace(DubinsCarStateSpace):
    """TODO: modify the dynamics and/or simulation here to include perturbations"""
    def __init__(self, dubins : DubinsCarStateSpace):
        super().__init__(dubins.t_step, dubins.car_L, dubins.car_w, dubins.a_bound, dubins.psi_bound, 
                    dubins.v_bound, dubins.phi_bound)

    def dynamics(self, state: list, control: list) -> list:
        return super().dynamics(state, control)
        
    def step(self, state : list, control : list, t_step=None) -> list:
        return super().step(state, control, t_step)

    def integrate(self, state: list, control: list, T_total : float) -> list:
        return super().integrate(state, control, T_total)

    def rollout(self, state: list, policy: callable, T_total : float, t_start : float=0.0) -> list:
        return super().rollout(state, policy, T_total, t_start)



def plot_trajs(ts,xs,us,ax, **kwargs):
    assert len(ts) == len(xs)
    assert len(ts) == len(us)+1
    ax.plot(ts,[x[0] for x in xs],label='x',**kwargs)
    ax.plot(ts,[x[1] for x in xs],label='y',**kwargs)
    ax.plot(ts,[np.mod(x[2]+np.pi,np.pi*2)-np.pi for x in xs],label='theta',**kwargs)
    ax.plot(ts,[x[3] for x in xs],label='v',**kwargs)
    ax.plot(ts,[x[4] for x in xs],label='phi',**kwargs)
    ax.plot(ts[:-1],[u[0] for u in us],label='a',**kwargs)
    ax.plot(ts[:-1],[u[1] for u in us],label='psi',**kwargs)
    # ax.legend()


def traj_opt_direct_transcription_t(dubins : DubinsCarCSpace,
                        x0 : np.array, xtarget : np.array,
                        T : int,
                        dt : float,
                        x_initial : Union[str,List[np.array]]='x0',
                        u_initial : Union[str,List[np.array]]='random',
                        weights = [3,0.02,5],
                        safe_dis = 0.2,
                        ab=[2,4],
                        bfunction = 1
                        ) -> Tuple[List[np.array],List[np.array],float]:
    assert len(x0) == 5
    assert len(xtarget) == 5
    assert T >= 1
    assert dt > 0
    
    #this defines the simulation function, in casadi autodiff form
    x = casadi.SX.sym('x',5)
    u = casadi.SX.sym('u',2)


    def sim_dubins(x,u):
        px,py,theta,v,phi = [x[i] for i in range(5)]
        a,psi = [u[i] for i in range(2)]
        dx = casadi.cos(theta)*v
        dy = casadi.sin(theta)*v
        dtheta = v*casadi.tan(phi)/dubins.dynamics.car_L
        dphi = psi
        dv = a
        dx = casadi.vertcat(dx,dy,dtheta,dv,dphi)
        return [x+dt*dx]
    simfunc_dubins_2nd_order = casadi.Function('f',[x,u],sim_dubins(x,u))
    
    #set the initial state and control trajectories
    if isinstance(x_initial,str):
        if x_initial == 'x0':
            x_initial = [x0]*(T+1)
        elif x_initial == 'line':
            x_initial = [None]*(T+1)
            for i in range(T+1): 
                s = float(i) / T
                x_initial[i] = x0+s*(xtarget-x0)
        elif x_initial == 'random':
            x_initial = [None]*(T+1)
            for i in range(T+1): 
                x_initial[i] = np.array([np.random.uniform(*dubins.x_bound),np.random.uniform(*dubins.y_bound)\
                             ,np.random.uniform(0,2*np.pi),np.random.uniform(*dubins.dynamics.v_bound),\
                                np.random.uniform(*dubins.dynamics.phi_bound)])
    assert len(x_initial) == T+1
    if isinstance(u_initial,str):
        if u_initial == 'zero':
            u_initial = [np.array([0.0,0.0])]*T
        elif u_initial == 'random':
            u_initial = [None]*T
            for i in range(T):
                u_initial[i] = np.array([np.random.uniform(*dubins.dynamics.a_bound),np.random.uniform(*dubins.dynamics.psi_bound)])
    assert len(u_initial) == T
    
    #set up the optimization variables, including T+1 state variables and T control variables
    opti = casadi.Opti()
    xtraj = opti.variable(T+1,5)
    utraj = opti.variable(T,2)

    def cal_k(dis,ab):
        k=[]
        k.append(np.exp(dis[0]**ab[0]))
        k.append(np.exp(-dis[1]**ab[1]))
        return k
    def terminal_cost(x):
        return weights[0]*((x[0]-xtarget[0])**2+(x[1]-xtarget[1])**2 + (x[2]-xtarget[2])**2 + x[3]**2)
    def control_cost(u,i):
        #TODO: consider control costs here
        control_cost = u[0]**2+u[1]**2 
        return weights[1]*control_cost
    def state_cost_inv(x, epsilon = 1e-6,d_safe=safe_dis):
        #TODO: set up obstacle avoidance costs here
        state_cost=0
        # k=[10,0.2]
        for obs in dubins.obstacles_points:
            # Compute cost for each sampled point
            x_o, y_o=obs[0],obs[1]
            dist_sq = casadi.sqrt((x[0] - x_o)**2 + (x[1] - y_o)**2+epsilon)
            dist_sq = casadi.fmax(epsilon, dist_sq-d_safe-dubins.obstacle_radius)
            state_cost += 1/dist_sq
        return weights[2]*state_cost
    


    def state_cost_novel(x, epsilon = 1e-6,d_safe=safe_dis):
        #TODO: set up obstacle avoidance costs here
        state_cost_=0
        d_safe = d_safe+dubins.obstacle_radius
        k=cal_k((np.array([dubins.dynamics.car_L,dubins.dynamics.car_w])+np.array([d_safe,d_safe*0.25]))*0.8,ab)
        for obs in dubins.obstacles_points:
            # Compute cost for each sampled point
            x_o, y_o=obs[0],obs[1]
            # dist_sq = casadi.sqrt((x[0] - x_o)**2 + (x[1] - y_o)**2)
            delta_x = ca.fabs(x[0]-x_o+epsilon)
            delta_y = ca.fabs(x[1]-y_o+epsilon)
            # delta_x = ca.sqrt((x[0] - x_o)**2+epsilon)*ca.cos(x[3])+ca.sqrt((x[1] - y_o)**2+epsilon)*ca.sin(x[3])
            # delta_y = ca.sqrt((ca.sqrt((x[0] - x_o)**2+epsilon)*ca.sin(x[3])-ca.sqrt((x[1] - y_o)**2+epsilon)*ca.cos(x[3]))**2+epsilon)
            # dist_sq = casadi.fmax(epsilon, dist_sq-d_safe-dubins.obstacle_radius)
            # delta_x =ca.fmin(delta_x,3)
            # delta_y =ca.fmin(delta_y,3)
            denominator_x = casadi.exp((delta_x+epsilon)**ab[0]) + k[0]
            denominator_y = casadi.exp(-(delta_y+epsilon)**ab[1]) + k[1]
            # state_cost += k[0]/(casadi.exp(delta_x**ab[0])+k[0])*\
            #     (1-k[1]/(casadi.exp(-delta_y**ab[1])+k[1]))
            state_cost_ += k[0] / denominator_x * (1 - k[1] / denominator_y)
            # state_cost += casadi.exp(delta_x**0.9)
        return weights[3]*state_cost_
    if bfunction == 1:
        state_cost = state_cost_inv
    else:
        state_cost = state_cost_novel
    def running_cost(x,u,i):
        return control_cost(u,i) + state_cost(x)
    #setup objective function
    obj = terminal_cost(xtraj[T,:])
    for i in range(T):
        obj = obj + dt*running_cost(xtraj[i,:],utraj[i,:],i)
    opti.minimize(obj)
    #initial state
    opti.subject_to(xtraj[0,:]==x0.reshape((1,5)))
    for i in range(T):
        opti.subject_to(xtraj[i+1,:]==simfunc_dubins_2nd_order(xtraj[i,:],utraj[i,:]).T)
    #state and control bounds
    for i in range(T):
        opti.subject_to(opti.bounded(dubins.x_bound[0],xtraj[i+1,0],dubins.x_bound[1]))
        opti.subject_to(opti.bounded(dubins.y_bound[0],xtraj[i+1,1],dubins.y_bound[1]))
        opti.subject_to(opti.bounded(dubins.dynamics.v_bound[0],xtraj[i+1,3],dubins.dynamics.v_bound[1]))
        opti.subject_to(opti.bounded(dubins.dynamics.phi_bound[0],xtraj[i+1,4],dubins.dynamics.phi_bound[1]))
        opti.subject_to(opti.bounded(dubins.dynamics.a_bound[0],utraj[i,0],dubins.dynamics.a_bound[1]))
        opti.subject_to(opti.bounded(dubins.dynamics.psi_bound[0],utraj[i,1],dubins.dynamics.psi_bound[1]))
    p_opts = {"expand":True}
    s_opts = {"max_iter":1000, "print_level": 0, "sb": "yes", "print_timing_statistics": "no"
              }
    # Save the current stdout
    original_stdout = sys.stdout

    # Redirect stdout to null
    sys.stdout = open(os.devnull, 'w')
    opti.solver("ipopt",p_opts,s_opts)
    
    opti.set_initial(xtraj[0,:],x_initial[0])
    for i in range(T):
        opti.set_initial(xtraj[i+1,:],x_initial[i])
        opti.set_initial(utraj[i,:],np.asarray(u_initial[i]))
    t0 = time.time()
    # sol = opti.solve()
    
    # try:
    sol = opti.solve()
    sys.stdout = original_stdout
    t1 = time.time()
    #extract the solution
    xt = sol.value(xtraj)
    ut = sol.value(utraj)
    # for i in range(T):
    #     # mx=casadi.vertcat(*[xtraj[i,j] for j in range(5)])
    #     mx = ca.MX.sym('mx',5)
    #     print("\nmx:",mx)
    #     f_debug = casadi.Function("f_debug",[mx], [state_cost(mx)])
    #     print(f"{i}th value {f_debug(xt[i,:])}")
    # print("Terminal cost:",'%.3f'%terminal_cost(xt[T,:]))
    # print("Control costs:",', '.join(['%.3f'%float(control_cost(ut[i,:],i)) for i in range(T)]))
    # print("State costs: ",', '.join(['%.3f'%float(state_cost(xt[i,:],i)) for i in range(T+1)]))
    return (xt,ut,t1-t0)
    # except Exception as e:
    #     sys.stdout = original_stdout
    #     print(f"\n\n error state:")
    #     variables = [x,u]
    #     for var in variables:
    #         print(f"{var}:{opti.debug.value(var)}")
    #     t1 = time.time()
    #     print("Solved in",t1-t0,"s")
    #     return ([0,0,0,0,0],[0,0],t1-t0)
    

def problem5():
    dubins,centres_ = setup_system_circular()
    dubins.x_bound = [-20.0, 20.0]
    dubins.y_bound = [-20.0, 20.0]
    sim_system = PerturbedDubinsCarStateSpace(dubins.dynamics)
    sim_system.t_step = 0.01  #finer time step for simulation

    x0 = np.array([0]*5)
    xtarget = np.array([0,15,np.pi/2,0,0])
    T_horizon = 10
    dt_optimizer = 0.1
    optimization_called_time=0
    n_substeps = 10
    T_total = 4
    PLOT_ANIM = True
    if PLOT_ANIM:
        import matplotlib.pyplot as plt
        plt.ion()
        fig,axs = plt.subplots(1,2,figsize=(10,4))
    def check_collision(trajectory):
        """
        Check if the optimized trajectory collides with any obstacles.
        """
        count = 0
        for point in trajectory:
            for obs in dubins.obstacles_points:
                if np.linalg.norm(point[:2] - obs[:2]) <= dubins.obstacle_radius:  # Collision detection
                    count+=1
        return count
    T = 0
    ts = [0]
    xs = [x0]
    us = []
    iters = 0
    plan_times = []
    plans = []
    t_optimizer = []
    collision_count=0
    m = 0
    while T < T_total:
        if iters % n_substeps == 0:
            optimization_called_time+=1
            plan_times.append(T)
            xt,ut,delta_t = traj_opt_direct_transcription_t(dubins,xs[-1],xtarget,T_horizon,
                                                               dt_optimizer,'line','zero',
                                                               weights = [3,0.02,1500],
                                                               dis=0.3,ab=[2,2])
            plans.append((xt,ut))
            t_optimizer.append(delta_t)
            if PLOT_ANIM:
                axs[0].clear()
                axs[1].clear()
                import shapely.plotting
                for obs in dubins.obstacles:
                    shapely.plotting.plot_polygon(obs, axs[0], color='black', add_points=False, alpha=0.5)
                # plot goal configuration
                axs[0].arrow(xtarget[0], xtarget[1], 1.0*np.cos(xtarget[2]), 
                    1.0*np.sin(xtarget[2]), color='red', width=.15, zorder=1e4)
                axs[0].plot([x[0] for x in xt],[x[1] for x in xt],linestyle=':',label='Plan %d'%(len(plans)))
                axs[0].plot([x[0] for x in xs],[x[1] for x in xs],label='Rollout')
                axs[0].axis('equal')
                axs[0].legend()
                plot_trajs(ts,xs,us,axs[1])
                if check_collision(xt)>0:
                    collision_count +=1
                fig.canvas.draw()
                fig.canvas.flush_events()
            if check_collision(xt)>0:
                collision_count +=1
 
        
        #modify the target
        xtarget[0] -= sim_system.t_step

        #simulate the first control
        T += sim_system.t_step
        us.append(ut[0])
        u = ut[0]
        u[0] = np.clip(u[0], *dubins.dynamics.a_bound)
        u[1] = np.clip(u[1], *dubins.dynamics.psi_bound)
        xnext = sim_system.step(xs[-1],u,sim_system.t_step)
        xs.append(np.array(xnext))
        ts.append(T)
        iters += 1
    if PLOT_ANIM:
        plt.ioff()
    t_sum = np.sum(np.array(t_optimizer))
    t_mean = np.mean(np.array(t_optimizer))
    print(f"collision time: {collision_count}")
    collision_count = collision_count/optimization_called_time
    collision_rate_final = check_collision(xs)
    print(f"optimized step collision: {collision_count}")
    print(f"final trajectory collision rate {collision_rate_final}")
    print(f"Total oprimization time {t_sum}")
    print(f"Average oprimization time {t_mean}")

    fig,axs = plt.subplots(1,2,figsize=(10,4))
    
    for obs in dubins.obstacles:
        shapely.plotting.plot_polygon(obs, axs[0], color='black', add_points=False, alpha=0.5)
    for i,(xt,ut) in enumerate(plans):
        if i%5 == 0:
            axs[0].plot([x[0] for x in xt],[x[1] for x in xt],linestyle=':',label='Plan %d'%(i+1))
    axs[0].plot([x[0] for x in xs],[x[1] for x in xs],label='Rollout')
    axs[0].axis('equal')
    axs[0].legend()
    plot_trajs(ts,xs,us,axs[1])
    plt.show()
    return 0

def plot_obstacle(dubins:DubinsCarCSpace,axs:matplotlib.axes.Axes):
    for obs in dubins.obstacles:
        shapely.plotting.plot_polygon(obs, axs , color='black', add_points=False, alpha=0.5)

def display_obs():
    dubins,centres_ = setup_system_circular()
    fig, axes = plt.subplots(2,5)
    for i,obs in enumerate(centres_):
        r_1=obs[0]
        obs = np.array(obs[1:],dtype=float)
        obstacles_Polygon,obstacles=generate_polygon(obs, r_1)
        #TODO: you may want to set up obstacles here
        dubins.obstacles_points = obs
        dubins.obstacles = obstacles_Polygon
        dubins.obstacle_radius = r_1
        axes[i//5][i%5].set_xlim(-10, 10)     # Set x-axis limits
        axes[i//5][i%5].set_ylim(-3, 10)    
        axes[i//5][i%5].set_aspect('equal')
        axes[i//5][i%5].set_title(f'centre type:{i%5}, radius:{dubins.obstacle_radius}')
        plot_obstacle(dubins,axes[i//5][i%5])
    plt.show()


def plot_trajectories(dubins, ax, x):
    # Plot trajectories
    vehicle = dubins.create_vehicle_polygon(x[0], x[1], x[2])
    x_veh, y_veh = vehicle.exterior.xy
    ax.plot(x_veh, y_veh, "g", alpha=0.6)

def test_with_obs():
    plt.clf()
    dubins,centres_ = setup_system_circular()
    dubins.x_bound = [-20.0, 20.0]
    dubins.y_bound = [-20.0, 20.0]
    sim_system = PerturbedDubinsCarStateSpace(dubins.dynamics)
    sim_system.t_step = 0.01  #finer time step for simulation
    prefix = "collosion_rates"
    prefix2 = "optimization_time"
    for bf in [0,1]:
        collision_name = "_".join([prefix,str(bf)])
        optimization_name = "_".join([prefix2,str(bf)])
        globals()[collision_name] = []
        globals()[optimization_name] = []
        fig,axs = plt.subplots(2,6)
        for i,obs in tqdm(enumerate(centres_),total = len(centres_),desc=f"Processing {bf}"):
            r_1=obs[0]
            obs = np.array(obs[1:],dtype=float)
            obstacles_Polygon,obstacles=generate_polygon(obs, r_1)
            dubins.obstacles_points = obs
            dubins.obstacles = obstacles_Polygon
            dubins.obstacle_radius = r_1
            x0 = np.array([0]*5)
            xtarget = np.array([0,15,np.pi/2,0,0])
            T_horizon = 10
            dt_optimizer = 0.1
            optimization_called_time=0
            n_substeps = 10
            T_total = 5
            # PLOT_ANIM = True
            # if PLOT_ANIM:
            #     import matplotlib.pyplot as plt
            #     plt.ion()
            #     fig,axs = plt.subplots(1,2,figsize=(10,4))
            def check_collision(trajectory):
                """
                Check if the optimized trajectory collides with any obstacles.
                """
                count = 0
                for point in trajectory:
                    for obs in dubins.obstacles_points:
                        if np.linalg.norm(point[:2] - obs[:2]) <= dubins.obstacle_radius:  # Collision detection
                            count+=1
                return count
            T = 0
            ts = [0]
            xs = [x0]
            us = []
            iters = 0
            plan_times = []
            plans = []
            t_optimizer = []
            collision_count=0
            m = 0
            while T < T_total:
                if iters % n_substeps == 0:
                    optimization_called_time+=1
                    plan_times.append(T)
                    xt,ut,delta_t = traj_opt_direct_transcription_t(dubins,xs[-1],xtarget,T_horizon,
                                                                    dt_optimizer,'line','random',
                                                                    weights = [3,0.02,10,1200],
                                                                    safe_dis=0.8,ab=[2,1.5],bfunction = bf)
                    plans.append((xt,ut))
                    t_optimizer.append(delta_t)
                    # if PLOT_ANIM:
                    #     axs[0].clear()
                    #     axs[1].clear()
                    #     import shapely.plotting
                    #     for obs in dubins.obstacles:
                    #         shapely.plotting.plot_polygon(obs, axs[0], color='black', add_points=False, alpha=0.5)
                    #     # plot goal configuration
                    #     axs[0].arrow(xtarget[0], xtarget[1], 1.0*np.cos(xtarget[2]), 
                    #         1.0*np.sin(xtarget[2]), color='red', width=.15, zorder=1e4)
                    #     axs[0].plot([x[0] for x in xt],[x[1] for x in xt],linestyle=':',label='Plan %d'%(len(plans)))
                    #     axs[0].plot([x[0] for x in xs],[x[1] for x in xs],label='Rollout')
                    #     axs[0].axis('equal')
                    #     axs[0].legend()
                    #     plot_trajs(ts,xs,us,axs[1])
                    #     fig.canvas.draw()
                    #     fig.canvas.flush_events()
                    


                    if check_collision(xt)>0:
                        collision_count +=1
                    if check_collision(xt)>0:
                        collision_count +=1
                #modify the target
                # xtarget[0] -= sim_system.t_step

                #simulate the first control
                T += sim_system.t_step
                us.append(ut[0])
                u = ut[0]
                u[0] = np.clip(u[0], *dubins.dynamics.a_bound)
                u[1] = np.clip(u[1], *dubins.dynamics.psi_bound)
                xnext = sim_system.step(xs[-1],u,sim_system.t_step)
                xs.append(np.array(xnext))
                ts.append(T)
                iters += 1
            # if PLOT_ANIM:
            #     plt.ioff()
            t_sum = np.sum(np.array(t_optimizer))
            t_mean = np.mean(np.array(t_optimizer))
            # print(f"collision time: {collision_count}")
            # collision_count = collision_count/optimization_called_time
            # collision_rate_final = check_collision(xs)/T_total

            # print(f"optimized step collision rate: {collision_count}")
            # print(f"final trajectory collision rate {collision_rate_final}")
            # print(f"Total oprimization time {t_sum}")
            # print(f"Average oprimization time {t_mean}")
            # globals()[collision_name].append(collision_rate_final)
            globals()[optimization_name].append(t_mean)
            
            
            for obs_ in dubins.obstacles:
                shapely.plotting.plot_polygon(obs_, axs[i//6][i%6], color='black', add_points=False, alpha=0.5)
            for m,(xt,ut) in enumerate(plans):
                if m%5 == 0:
                    axs[i//6][i%6].plot([x[0] for x in xt],[x[1] for x in xt],linestyle=':',label='Plan %d'%(m+1))
            axs[i//6][i%6].plot([x[0] for x in xs],[x[1] for x in xs],label='Rollout')
            axs[i//6][i%6].arrow(xtarget[0], xtarget[1], 1.0*np.cos(xtarget[2]), 
                1.0*np.sin(xtarget[2]), color='red', width=.16, zorder=1e4)
            axs[i//6][i%6].axis('equal')
            axs[i//6][i%6].set_title(f'centre type:{i%6}, radius:{dubins.obstacle_radius}')
            # axs[i//6][i%6].legend()
            # plot_trajs(ts,xs,us,axs[1])
            plot_trajectories(dubins,axs[i//6][i%6],xs[0])
            plot_trajectories(dubins,axs[i//6][i%6],xs[-1])
        if bf == 1:
            fig.suptitle("Inverse proportional", fontsize=16)
        else:
            fig.suptitle("Novel barrier function", fontsize=16)
        plt.show(block=False)
    fig, axs = plt.subplots(1,1)
    ts = range(len(optimization_time_0))
    # axs[0].plot(ts,collosion_rates_0,label="novel")
    # axs[0].plot(ts,collosion_rates_1,label="inverse")
    axs.plot(ts,optimization_time_0,label = "novel")
    axs.plot(ts,optimization_time_1,label = "inverse")
    axs.set_xlabel("Obstacles setting")
    axs.set_ylabel("TIem consumed (s)")
    axs.set_title("Comparision of optimization time")
    plt.legend()
    # plt.ioff()
    plt.show()
    
    return 0






if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        PROBLEM = sys.argv[1]
    # PROBLEM = '1A'
    PROBLEM = '1'
    if PROBLEM.startswith('1'):
        display_obs()
        print(1)
    if PROBLEM.startswith('2') or PROBLEM.startswith('3'):
        _=test_with_obs()
        print(2)
    if PROBLEM.startswith('4'):
        print(3)
        # problem1C_()
    if PROBLEM.startswith('t'):
        _ = problem5()

