# Eulerian-Fluid-Simulation

I have implemented an Eulerian Fluid Simulation where grid is represented by a staggered MAC Grid. 

# Implementation Details 
 - Implementation is done with Taichi
 - Velocity field lies on the cell faces
 - Pressure and Quantity (dye in this case) fields lie in the cell centers.
 - Pressure is currently computed using Jacobi Iterations
 
## The Local and World Grid Logic
It is hard to find people explaining the grid logic and index conversion they used in their code. I do not prefer converting indices in place in the code without explicitly stating in which grid the computation is done. The allocated fields are stored in row-major order whose origin is at the top-left. The cell centers are used as the cell indices. In other words, I use the classical multi-dimensional array convention. I did not want to do the math in this grid. Thus, I created a canonical grid which I call the world grid. 


### Properties of the World Grid
  - The origin lies at the top-left. The cell centers are used as the cell indices. 
  - Fields are stored in this grid.
  - Sampling is done in this field.

### Properties of the World Grid
  - The origin lies at the bottom-left. The cell centers are used as the cell indices. 
  - The math, like derivative computation and time integration, is done in this grid as it is more natural to me.
  - This grid serves as the canonical domain for all of the grids. As seen in the code, see advection functions, while converting indices from one field to the other; this grid is used as the intermediate domain. The idea resembles the similarity transformation from Linear Algebra.
  - This grid is purely conceptual. It is not stored in the memory.
  
 
 
# Some Results

## Different Time Integration Schemes
I implemented RK-1, RK-2 and RK-3 time integrators. Below, the comparison between them can be seen.

<table><tr>
<td> 
  <p align="center" style="padding: 10px">
    <video src="https://user-images.githubusercontent.com/44121631/223439086-5122cc6c-c43f-4591-800e-f7beaf0d47cc.mov" width="320">
    <br>
  </p> 
  <p align="center">
  <strong>RK-1</strong>
  </p>
</td>
<td> 
  <p align="center" style="padding: 10px">
    <video src="https://user-images.githubusercontent.com/44121631/223444740-4649fd6e-660c-40b5-bf08-47064dc31ebd.mov" width="320">
    <br>
  </p> 
  <p align="center">
  <strong>RK-2</strong>
  </p>
</td>
<td> 
  <p align="center" style="padding: 10px">
    <video src="https://user-images.githubusercontent.com/44121631/223445163-ad11b2a7-7835-4a45-9601-64e034258055.mov" width="320">
    <br>
  </p> 
  <p align="center">
  <strong>RK-3</strong>
  </p>
</td>
</tr></table>



