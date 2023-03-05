import taichi as ti


class DoubleBuffer:
  """
  Class to use double buffering. Cur is the last simulated frame. Each time we will read its values to simulate into the next buffer.
  At the we will swap and render the cur. So, the end values are placed in the cur buffer back again.
  Operation:
  simulate(cur, next) : read from cur and simulate into next
  swap()
  render(cur): cur is always the frame we will be rendering. At the end it becomes the last frame for the next timestep.
  
  Note that between consecutive simulation function we need swap since we read from cur and write onto next. To use the result of the previous
  step we need a swap
  
  For example:
  self_advection(v.cur, v.next)
  v.swap()
  solve_incompressibility(v.cur, v.next)
  v.swap()
  """
  def __init__(self, cur, next):
    self.cur = cur
    self.next = next

  def swap(self):
    self.cur, self.next = self.next, self.cur



@ti.func
def local_to_world_grid(i: ti.template(), j: ti.template(), shape: ti.template(), h: ti.template(), offset: ti.template()):
    """
    Parameters:
    -----------
      j
    -----
 i  | . |
    | . |
    -----
    i: row index
    j: column index 
    shape: shape of the local grid (n_y, n_x) (basically number of cells in the each axis) 
    h: grid spacing in the world space (in local grid spacing is 1, integer coordinates are evenly spaced)
    offset: (x, y) offset to correct the place we have landed on. Offset depends on the grid we are sampling from 
    if index lies at the center: offset = (0.0, 0.0)
    if index lies at the vertical walls of the cells = (-h/2, 0.0) (correct the x coord in the world)
    if index lies at the horizontal walls of the cells = (0.0, h/2) (correct the y coord in the world)
    Returns:
    --------
    x, y: x, y coordinates in the world grid


    Note:
    -----
    Note that the shape of the grid is measured with number of cells, not the number of values we store
    In other words, when using the function to retrieve the mapping in a staggered value (like velocity),
    we will still supply the shape of the centered grid since it has the dimension (n_y, n_x).
    It might be strange but this is because we apply the mapping depending on the number of cells not the
    number of indices.
    """
    x, y = j*h, (shape[0]-1-i)*h
    return x + offset[0], y + offset[1]


@ti.func
def world_to_local_grid(x: ti.template(), y: ti.template(), shape: ti.template(), h: ti.template(), offset: ti.template()):
    """
    Parameters:
    -----------
    
    -----
    | . |
    | . |
  y | . |
      x
    -----
    x: x coordinate
    y: y coordinate
    shape: shape of the local grid (n_y, n_x) (basically number of cells in the each axis) 
    h: grid spacing in the world space (in local grid spacing is 1, integer coordinates are evenly spaced)
    offset: (i, j) offset to correct the place we have landed on. Offset depends on the grid we are sampling from 
    if index lies at the center: offset = (0.0, 0.0)
    if index lies at the vertical walls of the cells = (0.0, 0.5) (correct the j coord in the local)
    if index lies at the horizontal walls of the cells = (0.5, 0.0) (correct the i coord in the local)
    Returns:
    --------
    i, j: i and j indices of the local grid. Note that i and j are clamped so it is guaranteed that they
    lie inside the grid 
    
    Note:
    -----
    Note that the shape of the grid is measured with number of cells, not the number of values we store
    In other words, when using the function to retrieve the mapping in a staggered value (like velocity),
    we will still supply the shape of the centered grid since it has the dimension (n_y, n_x).
    It might be strange but this is because we apply the mapping depending on the number of cells not the
    number of indices.
    """
    i, j = shape[0]-1-(y/h) + offset[0], x/h + offset[1] 
    #clamp 
    #TODO: Right limit should be adjusted wrt centered and staggered grid. Not that much important for now
    i, j = ti.math.clamp(i, 0, shape[0]-1), ti.math.clamp(j, 0, shape[1]-1)
    
    return i, j 


@ti.func
def sample(f: ti.template(), p: ti.template()) -> ti.f32:
    """
    Samples a value from the field using bilinear interpolation. Note that sampling is done in the local grid.
    Parameters:
    -----------
    f: field to sample from
    p: p is the point inside the local grid. p = [i, j]. Note that i and j are floats 

    Returns:
    --------
    The sampled value
    """
    i, j = ti.cast(ti.floor(p[0]), dtype=ti.i32), ti.cast(ti.floor(p[1]), dtype=ti.i32) 
    ip, jp = ti.min(i+1, f.shape[0]-1), ti.min(j+1, f.shape[1]-1)
    t, s = p[0] - i, p[1] - j
    return (f[i, j] * (1-s) + f[i, jp] * s) * (1-t) + (f[ip, j] * (1-s) + f[ip, jp] * s) * t

