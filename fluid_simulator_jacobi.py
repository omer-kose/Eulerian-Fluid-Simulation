import taichi as ti


import utils
from utils import local_to_world_grid, world_to_local_grid, sample

ti.init(arch=ti.gpu) 

#Local Grid Properties (Note that world grid is automatically defined with the properties we defined)
n_x = 512 #Number of cells in the x axis (j axis actually)
n_y = 512 #Number of cells in the y axis (i axis actually)
num_cells = (n_y, n_x) # (num_cells_in_y, num_cells_in_x)
h = 1 #Grid Spacing
#Grid Fields
p = ti.field(dtype=ti.f32, shape=(n_y, n_x)) #Pressure
p_second = ti.field(dtype=ti.f32, shape=(n_y, n_x)) #Second Pressure Buffer
u = ti.field(dtype=ti.f32, shape=(n_y, n_x+1)) #U component of the velocity
u_second = ti.field(dtype=ti.f32, shape=(n_y, n_x+1)) #Second U Buffer
v = ti.field(dtype=ti.f32, shape=(n_y+1, n_x)) #V component of the velocity
v_second = ti.field(dtype=ti.f32, shape=(n_y+1, n_x)) #Second V Buffer
dye = ti.field(dtype=ti.f32, shape=(n_y, n_x)) #Dye color
dye_second = ti.field(dtype=ti.f32, shape=(n_y, n_x)) #Second Dye Buffer
div = ti.field(dtype=ti.f32, shape=(n_y, n_x)) #Divergence field
#Double Buffers
p_buffers = utils.DoubleBuffer(p, p_second)
u_buffers = utils.DoubleBuffer(u, u_second)
v_buffers = utils.DoubleBuffer(v, v_second)
dye_buffers = utils.DoubleBuffer(dye, dye_second)





"""
Constant offsets to map between the grids
Naming Convention: FROM_TO_TARGET_X: From grid to Target grid where the values lie at X.
FROM and TARGET: local (l), and world (w)
X: C, U, V 
C: value lies at the cell centers
U: value lies at the vertical walls (u for u component of the velocity)
V: value lies at the horizonal walls (v for v component of the velocity)
"""
#Local To World Grid
L_TO_W_C_OFFSET = ti.Vector([0.0, 0.0])
L_TO_W_U_OFFSET = ti.Vector([-h/2, 0.0])
L_TO_W_V_OFFSET = ti.Vector([0.0, h/2])
#World to Local Grid
W_TO_L_C_OFFSET = ti.Vector([0.0, 0.0])
W_TO_L_U_OFFSET = ti.Vector([0.0, 0.5])
W_TO_L_V_OFFSET = ti.Vector([0.5, 0.0])



#Simulation Properties
dt = 0.02 #TODO: Make this adjustable depending on the max velocity and acceleration as described in Bridson and Müller
inv_dt = 1.0 / dt
density = 1.0 #Density of the fluid
inv_density = 1.0 / density
gravity = ti.Vector([0.0, -9.8])
RK = 1 #Runge-Kutta Integration Order (1-2-3)
#Jacobi Iteration Properties
max_iterations = 30 
allowable_error = 1e-3

#Rendering and GUI Properties
pause = False
num_substeps = 15




#Integrators
@ti.func
def RK_1(p: ti.template(), vel: ti.template(), dt: ti.template()):
  """
  Applies Order 1 Runge-Kutte time integration. 

  Parameters:
  ----------
  p: Position in the world grid
  vel: Current velocity there
  dt: Timestep 

  Returns:
  --------
  The integrated world pos
  """
  return p - dt * vel



@ti.func
def RK_2(p: ti.template(), vel: ti.template(), dt: ti.template(), uf: ti.template(), vf: ti.template()):
  """
  Applies Order 2 Runge-Kutte time integration. 

  Parameters:
  ----------
  p: Position in the world grid
  vel: Current velocity there
  dt: Timestep 
  uf: current u component field
  vf: current v component field
  Returns:
  --------
  The integrated world pos
  """
  mid_p = p - 0.5 * dt * vel
  #Sample the velocity at mid point
  u_i, u_j = world_to_local_grid(mid_p[0], mid_p[1], num_cells, h, W_TO_L_U_OFFSET)
  v_i, v_j = world_to_local_grid(mid_p[0], mid_p[1], num_cells, h, W_TO_L_V_OFFSET)
  sample_u = sample(uf, ti.Vector([u_i, u_j]))
  sample_v = sample(vf, ti.Vector([v_i, v_j]))
  
  return p - dt * ti.Vector([sample_u, sample_v]) 



@ti.func
def RK_3(p: ti.template(), vel: ti.template(), dt: ti.template(), uf: ti.template(), vf: ti.template()):
  """
  Applies Order 2 Runge-Kutte time integration. 

  Parameters:
  ----------
  p: Position in the world grid
  vel: Current velocity there
  dt: Timestep 
  uf: current u component field
  vf: current v component field
  Returns:
  --------
  The integrated world pos
  """
  vel_1 = vel
  p1 = p - 0.5 * dt * vel_1

  #Sample the velocity at p1
  u_i, u_j = world_to_local_grid(p1[0], p1[1], num_cells, h, W_TO_L_U_OFFSET)
  v_i, v_j = world_to_local_grid(p1[0], p1[1], num_cells, h, W_TO_L_V_OFFSET)
  vel_2 = ti.Vector([sample(uf, ti.Vector([u_i, u_j])), sample(vf, ti.Vector([v_i, v_j]))])
  p2 = p - 0.75 * dt * vel_2

  #Sample the velocity at p2
  u_i, u_j = world_to_local_grid(p2[0], p2[1], num_cells, h, W_TO_L_U_OFFSET)
  v_i, v_j = world_to_local_grid(p2[0], p2[1], num_cells, h, W_TO_L_V_OFFSET)
  vel_3 = ti.Vector([sample(uf, ti.Vector([u_i, u_j])), sample(vf, ti.Vector([v_i, v_j]))])

  return p - dt * (2.0 / 9.0 * vel_1 + 1.0 / 3.0 * vel_2 + 4.0 / 9.0 * vel_3)


@ti.kernel
def apply_gravity(vf: ti.template()):
    """
    Apply gravity to the velocity field directly. 
    Note: This directly applies to current buffer of the velocity. Thus, this is an in-place operation
    
    Parameters:
    -----------
    vf: velocity field (v component).
    """
    #Loop over the fluid 
    for i in range(1, vf.shape[0]-1):
        for j in range(1, vf.shape[1]-1):
            vf[i, j] += dt * gravity[1]     



   
@ti.kernel
def self_advection(uf: ti.template(), uf_new: ti.template(), vf: ti.template(), vf_new: ti.template()):
    """
    Applies semi-lagrangian self-advection.

    Parameters:
    -----------
    uf: current u velocity buffer to read from (cur)
    uf_new: next u velocity buffer to write onto (next)
    vf: current v velocity buffer to read from (cur)
    vf_new: next v velocity buffer to write onto (next)
    """
    #Loop over the fluid u components
    for i in range(1, uf.shape[0]-1):
        for j in range(1, uf.shape[1]-1):
            #Map (i, j) to the world grid
            x, y = local_to_world_grid(i, j, num_cells, h, L_TO_W_U_OFFSET)
            i_v, j_v = world_to_local_grid(x, y, num_cells, h, W_TO_L_V_OFFSET) #From world to V grid. Absolutely this is a mapping from u to v
            #Sample u and v components from the corresponding grids
            u_sample = uf[i, j]
            v_sample = sample(vf, ti.Vector([i_v, j_v]))
            backtraced_pos = ti.Vector([0.0, 0.0])
            if ti.static(RK == 1):
                backtraced_pos = RK_1(ti.Vector([x, y]), ti.Vector([u_sample, v_sample]), dt)
            elif ti.static(RK == 2):
                backtraced_pos = RK_2(ti.Vector([x, y]), ti.Vector([u_sample, v_sample]), dt, uf, vf)
            elif ti.static(RK == 3):
                backtraced_pos = RK_3(ti.Vector([x, y]), ti.Vector([u_sample, v_sample]), dt, uf, vf)
            
            #Now we have the backtraced position we can sample the u value there
            i_u, j_u = world_to_local_grid(backtraced_pos[0], backtraced_pos[1], num_cells, h, W_TO_L_U_OFFSET)
            uf_new[i, j] = sample(uf, ti.Vector([i_u, j_u]))            
            
    #Loop over the fluid v components
    for i in range(1, vf.shape[0]-1):
        for j in range(1, vf.shape[1]-1):
            #Map (i, j) to the world grid
            x, y = local_to_world_grid(i, j, num_cells, h, L_TO_W_V_OFFSET)
            i_u, j_u = world_to_local_grid(x, y, num_cells, h, W_TO_L_V_OFFSET) #From world to u grid. Absolutely this is a mapping from v to u
            #Sample u and v components from the corresponding grids
            v_sample = vf[i, j]
            u_sample = sample(uf, ti.Vector([i_u, j_u]))
            backtraced_pos = ti.Vector([0.0, 0.0])
            if ti.static(RK == 1):
                backtraced_pos = RK_1(ti.Vector([x, y]), ti.Vector([u_sample, v_sample]), dt)
            elif ti.static(RK == 2):
                backtraced_pos = RK_2(ti.Vector([x, y]), ti.Vector([u_sample, v_sample]), dt, uf, vf)
            elif ti.static(RK == 3):
                backtraced_pos = RK_3(ti.Vector([x, y]), ti.Vector([u_sample, v_sample]), dt, uf, vf)
            
            #Now we have the backtraced posiion we can sample the u value there
            i_v, j_v = world_to_local_grid(backtraced_pos[0], backtraced_pos[1], num_cells, h, W_TO_L_V_OFFSET)
            vf_new[i, j] = sample(vf, ti.Vector([i_v, j_v])) 


@ti.kernel
def advect_quantity(qf: ti.template(), qf_new: ti.template(), uf: ti.template(), vf: ti.template()):
    """
    Advects any quantity that lies in the grid centers using Semi-Lagrangian Advection.

    Parameters:
    -----------
    qf: Current quality field
    qf_new: Next quality field buffer to write onto
    uf: u component of the velocity field
    vf: v component of the velocity field
    """
    shape = qf.shape
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            #Map (i, j) to the world grid
            x, y = local_to_world_grid(i, j, num_cells, h, L_TO_W_C_OFFSET)
            #Sample u and v component there
            i_u, j_u = world_to_local_grid(x, y, num_cells, h, W_TO_L_U_OFFSET) #From world to U grid. 
            i_v, j_v = world_to_local_grid(x, y, num_cells, h, W_TO_L_V_OFFSET) #From world to V grid. 
            u_sample = sample(uf, ti.Vector([i_u, j_u]))
            v_sample = sample(vf, ti.Vector([i_v, j_v]))
            backtraced_pos = ti.Vector([0.0, 0.0])
            if ti.static(RK == 1):
                backtraced_pos = RK_1(ti.Vector([x, y]), ti.Vector([u_sample, v_sample]), dt)
            elif ti.static(RK == 2):
                backtraced_pos = RK_2(ti.Vector([x, y]), ti.Vector([u_sample, v_sample]), dt, uf, vf)
            elif ti.static(RK == 3):
                backtraced_pos = RK_3(ti.Vector([x, y]), ti.Vector([u_sample, v_sample]), dt, uf, vf)

            #Now we have the backtraced posiion we can sample the q value there
            i_q, j_q = world_to_local_grid(backtraced_pos[0], backtraced_pos[1], num_cells, h, W_TO_L_C_OFFSET)
            qf_new[i, j] = sample(qf, ti.Vector([i_q, j_q])) 


@ti.kernel
def p_handle_no_slip_boundary_condition(pf: ti.template()):
    """
    Handles no-slip boundary condition for pressures by setting pressure inside the solid walls to zero.
    Note that handling is done in-place since we do not need to read from the current buffer.
    Parameters: 
    -----------
    pf: pressure field  
    """
    #Set top and bottom solid pressures to zero
    shape = pf.shape
    for j in range(shape[1]):
        pf[0, j] = 0.0
        pf[shape[0]-1, j] = 0.0

    #Set left and right solid pressures to zero
    for i in range(shape[0]):
        pf[i, 0] = 0.0
        pf[i, shape[1]-1] = 0.0



@ti.kernel
def vel_handle_no_slip_boundary_condition(uf: ti.template(), vf: ti.template()):
    """
    Handles no-slip boundary condition for the velocity fields by extrapolating velocities into the solid
    walls. Note that like in the pressure boundary this operation done in-place

    Parameters:
    -----------
    uf: u component field
    vf: v component field
    """
    u_shape = uf.shape
    v_shape = vf.shape 
    
    #Set top and bottom vertical wall velocities
    for j in range(u_shape[1]):
        uf[0, j] = uf[1, j] 
        uf[u_shape[0]-1, j] = uf[u_shape[0]-2, j]

    #Set left and right horizontal wall velocities
    for i in range(v_shape[0]):
        vf[i, 0] = vf[i, 1]
        vf[i, v_shape[1]-1] = vf[i, v_shape[1]-2]
    

@ti.kernel
def compute_divergence(uf: ti.template(), vf: ti.template(), h: ti.template(), divf: ti.template()):
    """
    Compute the divergence field. Operation is done on divf in-place.

    Parameters:
    ----------
    uf: u component field
    vf: v component field
    h: world grid spacing (necessary for derivative computation)
    divf: divergence field
    """ 
    shape = divf.shape
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            divf[i, j] = ((uf[i, j+1] - uf[i, j]) / h) + ((vf[i, j] - vf[i, j+1]) / h) 




@ti.kernel
def p_jacobi_iteration(pf: ti.template(), pf_new: ti.template(), divf: ti.template()) -> ti.f32:
    """
    Computes pressures using Jacobi Iterations.
    
    Parameters:
    -----------
    pf: Current pressure field
    pf_new: New pressure field bufer to write onto
    divf: Divergence field
    """ 
    cum_p = 0.0 #Cumulative sum of the pressures in the current iteration
    cum_diff = 0.0 #Cumulative sum of the errors
    shape = pf.shape
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            pf_new[i, j] = 0.25 * (pf[i+1, j] + pf[i-1, j] + pf[i, j+1] + pf[i, j-1] - (density * h * h * inv_dt) * divf[i, j])
            p_diff = ti.abs(pf_new[i, j] - pf[i, j])
            cum_p += pf_new[i, j] * pf_new[i, j]
            cum_diff += p_diff * p_diff

    residual = ti.sqrt(cum_diff / cum_p)
    return residual



def solve_pressure_jacobi():
    """
    Solves for the pressures using Jacobi Iteration
    """
    residual = 1000
    num_iterations = 0
    while (residual > allowable_error and num_iterations < max_iterations):
        residual = p_jacobi_iteration(p_buffers.cur, p_buffers.next, div)
        p_buffers.swap()
        num_iterations += 1

    #At the end handle boundary conditions (we are not touching boundaries in iterations but anyway)
    p_handle_no_slip_boundary_condition(p_buffers.cur)


@ti.kernel
def projection(uf: ti.template(), uf_new: ti.template(), vf: ti.template(), vf_new: ti.template(), pf: ti.template()):
    """
    Applies the projection step which makes the velocity field divergence free.

    Parameters:
    ----------
    uf: Current u component of the field (u_buffers.cur)
    uf_new: New u buffer to write onto (u_buffers.next)
    vf: Current v component of the field (v_buffers.cur)
    vf_new: New v buffer to write onto (v_buffers.next)
    pf: Current pressure field
    """
    u_shape = uf.shape
    v_shape = vf.shape
    
    #Update u components
    for i in range(1, u_shape[0]-1):
        for j in range(1, u_shape[1]-1):
            uf_new[i, j] = uf[i, j] - dt * inv_density * ((pf[i, j] - pf[i, j-1]) / h) 


    #Update v components
    for i in range(1, v_shape[0]-1):
        for j in range(1, v_shape[1]-1):
            vf_new[i, j] = vf[i, j] - dt * inv_density * ((pf[i, j] - pf[i-1, j]) / h) 



def simulate():
    if not pause:
        for substep in range(num_substeps):
            #Self advection
            self_advection(u_buffers.cur, u_buffers.next, v_buffers.cur, v_buffers.next)
            u_buffers.swap()
            v_buffers.swap()
            #Quantity advection
            advect_quantity(dye_buffers.cur, dye_buffers.next, u_buffers.cur, v_buffers.cur)
            dye_buffers.swap()
            #External Forces
            apply_gravity(v_buffers.cur)
            #Handle boundary conditions
            vel_handle_no_slip_boundary_condition(u_buffers.cur, v_buffers.cur)
            #Projection 
            compute_divergence(u_buffers.cur, v_buffers.cur, h, div)
            solve_pressure_jacobi()
            projection(u_buffers.cur, u_buffers.next, v_buffers.cur, v_buffers.next, p_buffers.cur)
            u_buffers.swap()
            v_buffers.swap()
            #Handle boundary conditions
            vel_handle_no_slip_boundary_condition(u_buffers.cur, v_buffers.cur)




@ti.kernel
def init_simulation():
    for i in range(1, dye_buffers.cur.shape[0]-1):
        for j in range(1, dye_buffers.cur.shape[1]-1):
            if(i // 4 + j // 4 ) % 2 == 0:
                dye_buffers.cur[i, j] = 1.0



init_simulation()
gui = ti.GUI("Eulerian Fluid Simulation", res=(n_y, n_x))
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False 
        elif e.key == 'r':
            init_simulation()
        elif e.key == gui.SPACE:
            pause = not pause
    
    simulate()
    gui.set_image(dye_buffers.cur)
    gui.show()
