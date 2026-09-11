set term png
set output "radius.png"
beta = 4.06022                              # Scriven growth constant for dT = 1.25 K
alpha = 0.677/(958.4*4216.0)                # thermal diffusivity [m^2/s]
r0 = 5e-5                                   # initial bubble radius [m]
t0 = r0**2.0/(4.0*beta**2.0*alpha)          # virtual time origin [s]
set xran [0:0.002]
set grid
set xlabel "Time (s)"
set ylabel "Radius (m)"
set title "dT = 1.25 K"
plot 2.0*beta*sqrt(alpha*x) title "Theoretical", \
     "bench-data.dat" u ($1+t0):((3.0*$3/(4.0*pi))**(1.0/3.0)) w l title "Simulation"

