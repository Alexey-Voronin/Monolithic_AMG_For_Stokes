// DEALii https://github.com/dealii/dealii/blob/master/examples/step-49/example.geo
cl1 = 1;

Lx = 0.0;
Ux = 1.0;
Ly = 0.0;
Uy = 1.0;

Point(1) = {Lx, Uy, 0, 1};
Point(2) = {Ux, Uy, 0, 1};
Point(3) = {Lx, Ly, 0, 1};
Point(4) = {Ux, Ly, 0, 1};

// lines of the outer box:
Line(1) = {1, 2};
Line(2) = {4, 2};
Line(3) = {1, 3};
Line(4) = {3, 4};

Line Loop(5) = {1, -2, -4, -3};

// these define the boundary indicators in deal.II:
Physical Line(1) = {2};
Physical Line(0) = {3};
Physical Line(2) = {1, 4};

// you need the physical surface, because that is what deal.II reads in
Plane Surface(11) = {5};
Physical Surface(12) = {11};

Mesh.CharacteristicLengthFactor = CHAR_FACTOR;
//Mesh.Smoothing = 2;
