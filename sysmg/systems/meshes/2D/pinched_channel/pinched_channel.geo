/*
	Modified version of the .geo file originally proived by and described below

    Profile of the axisymmetric stenosis following a cosine
	function dependent on the axial coordinate x [1].
 
	 f(x) = R * (1 - f0/2 *(1 + Cos(2.0*Pi * (x-x0)/L) )
	 -- x0 maximum stenosis position.
	 -- L stenosis length.
	 -- R vessel radius
	 -- f0 obstruction fraction, ranging 0--1.
	 
	References:
 
	[1] Varghese, S. S., Frankel, S. H., Fischer, P. F., 'Direct numerical simulation of steotic flows. Part 1. Steady flow', Journal of Fluid Mechanics, vol. 582, pp. 253 - 280.
*/

ls = CHAR_FACTOR;

Xi = 100; // um
Xo = 100; // um
L = 100.0; // um

x0 = Xi + L/2.0;
R = 50.0;   // um
f0 = 0.5;   // 0--1

Z = 5;

Point(1) = {0, 0, 0, ls};
Point(2) = {Xi, 0, 0, ls};
Point(3) = {Xi, R, 0, ls};
Point(4) = {0, R, 0, ls};

Point(5) = {Xi + L, 0, 0, ls};
Point(6) = {Xi + L + Xo, 0, 0, ls};
Point(7) = {Xi + L + Xo, R, 0, ls};
Point(8) = {Xi + L, R, 0, ls};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

//	Line(9) = {2, 5};
// lower
pList[0] = 2; // First point label
nPoints = 21; // Number of discretization points (top-right point of the inlet region)
For i In {1 : nPoints}
x = Xi + L*i/(nPoints + 1);
pList[i] = newp;
Point(pList[i]) = {x,
( -1.0*R * (1 - f0/2.2 *(1 + Cos(2.0*Pi * (x-x0)/L) ) )+50.0),
0,
ls};
EndFor
pList[nPoints+1] = 5; // Last point label (top-left point of the outlet region)
Spline(newl) = pList[];

// upper
pList[0] = 3; // First point label
nPoints = 21; // Number of discretization points (top-right point of the inlet region)
For i In {1 : nPoints}
  x = Xi + L*i/(nPoints + 1);
  pList[i] = newp;
  Point(pList[i]) = {x,
                ( R * (1 - f0/2.2 *(1 + Cos(2.0*Pi * (x-x0)/L) ) )),
                0,
                ls};
EndFor
pList[nPoints+1] = 8; // Last point label (top-left point of the outlet region)
Spline(newl) = pList[];


Line Loop(11) = {4, 1, 2, 3};
Plane Surface(12) = {11};
Line Loop(13) = {2, 10, 8, -9};
Plane Surface(14) = {13};
Line Loop(15) = {8, 5, 6, 7};
Plane Surface(16) = {15};


Physical Surface(0) = {16, 14, 12};
Physical Line(0) = {4};
Physical Line(1) = {6};
Physical Line(2) = {1,3,7,5,9,10};
//Mesh.Smoothing = 5;
