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

Cx = Ux*1.8/5.;
Cy = Uy/2.0;
cr = 0.2;
Point(7) = {Cx,      Cy,     0, 1};
Point(8) = {Cx+cr,   Cy,     0, 1};
Point(9) = {Cx,      Cy+cr, -0, 1};
Point(10)= {Cx-cr,   Cy,     0, 1};
Point(11)= {Cx,      Cy-cr,  0, 1};

// lines of the outer box:
Line(1) = {1, 2};
Line(2) = {4, 2};
Line(3) = {1, 3};
Line(4) = {3, 4};

// the first cutout:
Ellipse(5) = {8, 7, 11, 9};
Ellipse(6) = {9, 7, 11, 10};
Ellipse(7) = {8, 7, 10, 11};
Ellipse(8) = {11, 7, 8, 10};

// loops of the outside and the two cutouts
Line Loop(9) = {1, -2, -4, -3};
Line Loop(10) = {5, 6, -8, -7};

// these define the boundary indicators in deal.II:
Physical Line(1) = {2};
Physical Line(0) = {3};
Physical Line(2) = {1, 4, 6, 5, 8, 7};
// Physical Line(2) = {1, 4};
// Physical Line(3) = {6, 5, 8, 7};


// you need the physical surface, because that is what deal.II reads in
Plane Surface(11) = {9, 10};
Physical Surface(12) = {11};

// some parameters for the meshing:
// Mesh.Algorithm = 8;
// Mesh.RecombineAll = 1;
 Mesh.CharacteristicLengthFactor = CHAR_FACTOR;
// Mesh.CharacteristicLengthFactor = 0.2;
// Mesh.SubdivisionAlgorithm = 1;
// Mesh.Smoothing = 5;
