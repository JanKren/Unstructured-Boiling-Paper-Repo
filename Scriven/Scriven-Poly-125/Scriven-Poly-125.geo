SetFactory("OpenCASCADE");

// Cubic domain: 0.3mm x 0.3mm x 0.3mm
// Centered at origin
Box(1) = {-0.00015, -0.00015, -0.00015, 0.0003, 0.0003, 0.0003};

// MeshSize = 2.4um gives ~125 cells per 0.3mm
// (compared to 4um in Dual which gives ~75 cells)
MeshSize {1:8} = 0.0000024;

Physical Surface("wall", 13) = {1:6};
Physical Volume("interior", 14) = {1};
