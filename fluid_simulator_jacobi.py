import taichi as ti


import utils
from utils import local_to_world_grid, world_to_local_grid, sample


ti.init(ti.gpu)

#Local Grid Properties (Note that world grid is automatically defined with the properties we defined)
n_x = 16 #Number of cells in the x axis (j axis actually)
n_y = 16 #Number of cells in the y axis (i axis actually)
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
gravity = ti.Vector([0.0, -9.8])
RK = 1 #Runge-Kutta Integration Order (1-2-3)

#Rendering Properties




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
            vf[i, j] += gravity[1]     



   
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
            
            #Now we have the backtraced position we can sample the u value there
            i_v, j_v = world_to_local_grid(backtraced_pos[0], backtraced_pos[1], num_cells, h, W_TO_L_V_OFFSET)
            vf_new[i, j] = sample(vf, ti.Vector([i_v, j_v])) 






self_advection(u_buffers.cur, u_buffers.next, v_buffers.cur, v_buffers.next)
u_buffers.swap()
v_buffers.swap()
print(u_buffers.cur)
