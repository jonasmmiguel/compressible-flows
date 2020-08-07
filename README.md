# Compressible Flows Workbench #

Helpful tools for modeling steady, non-reacting, single-phase, 1D flows where Mach number may surpass $M>0.3$ (significant compressibility effects). 

 ## Features ##

- Flow functions
  - Isentropic
  - Normal shock
  - Fanno
  - Rayleigh
- Simple handling of units (e.g. units conversion) 
- Automatic determination of friction factor $f(\varepsilon/D, Re_D)$ via Colebrook's equation
- Default properties of common gases

## Validity domain ##

- 1D flow modeling (gradients)
- Steady (no temporal variation) or quasi-steady (succession of states each in thermodynamic equilibreium) regimes
- Any Mach number $M \in [0, +\infty[$
- Negligible effects of 
  - chemical reactions
  - phase changes
  - gravitational potential changes
- Medium
  - Mono, bi- and some triatomic gases
  - any specific heat capacity and heat capacity ratio $(c_p, \gamma)$
  
## Extra: Interactive Visualization of Flow Functions ##
- [Isentropic](https://chart-studio.plotly.com/~jonasmmiguel/49/#/)
- [Normal Shock](https://chart-studio.plotly.com/~jonasmmiguel/51/#/)
- [Fanno](https://chart-studio.plotly.com/~jonasmmiguel/49/#/)
- [Rayleigh](https://chart-studio.plotly.com/~jonasmmiguel/55/#/)



