set grid
set xlabel "Time (s)"
set ylabel "Front position (m)"
plot "interface_position.num" using ($1+0.1):2 with lines title "T-Flows", \
     "interface_position.exa" using 1:2 with lines title "Analytical solution"
