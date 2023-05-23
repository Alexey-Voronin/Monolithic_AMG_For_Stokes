/* helpful tutorial to review before going through this file - 
 * https://bthierry.pages.math.cnrs.fr/tutorial/gmsh/occ/spline/
 */

SetFactory("OpenCASCADE");
Geometry.OCCAutoFix = 0;
h = CHAR_FACTOR;
Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;

/* Every (2D) object is defined above x-axis 
 * Rotational exctrusion is used to create the
 * final 3D object
 *
 *ymax ______   ______
 *    |  L0  \./  L1 |
 * 
 */

x0      = 0;  // left cylinder x-start
dx_L0   = 50;   // left cylinder length
dx_spln = 100;  // spline length
dx_L1   = 50;   // right cylinder length
ymax    = 30;   // max y-height 

// Pinched Part
x0_spln  = x0+dx_L0;
Point(0) = {x0_spln+dx_spln,        ymax,0,h};
Point(1) = {x0_spln,                ymax,0,h};

// Cylinders on the left and right sides of the pinch
Point(2) = {x0,    ymax,0,h};
x_end    = x0+dx_L0+dx_spln+dx_L1;
Point(3) = {x_end,ymax,0,h};
Line(2)  = {2,1};
Line(3)  = {0,3};

// Lines to close the cylinders at the ends
Point(4) = {x0,   0,0,h};
Point(5) = {x_end,0,0,h};
Line(4) = {2,4};
Line(5) = {3,5};

f0 = 0.9; // \in [0, 1] the smaller the f0 the wider the channel
pList[0] = 1; // First point label
nPoints = 21; // Number of discretization points (top-right point of the inlet region)
For i In {1 : nPoints}
  x = x0_spln + dx_spln*i/(nPoints + 1);
  pList[i] = newp;
       Point(pList[i]) = {x,
                      ymax-1*(ymax*f0)*(1-0.5*(1 + Cos(2.0*Pi * (x-x0_spln)/dx_spln))),
                     0, h};
EndFor
pList[nPoints+1] = 0; // Last point label 
Spline(1) = pList[];

// Rotate-Extrude
pinch = Extrude { {1,0,0}, {0,0,0}, 2*Pi} { Line{1} ; Layers{1}; Recombine;}; 
//Coherence; // remove duplicate lines
l_cyl = Extrude { {1,0,0}, {0,0,0}, 2*Pi} { Line{2} ; Layers{1}; Recombine;};
Coherence; // remove duplicate lines
l_cap = Extrude { {1,0,0}, {0,0,0}, 2*Pi} { Line{4} ; Layers{1}; Recombine;};
Coherence; // remove duplicate lines
r_cyl = Extrude { {1,0,0}, {0,0,0}, 2*Pi} { Line{3} ; Layers{1}; Recombine;};
Coherence; // remove duplicate lines
r_cap = Extrude { {1,0,0}, {0,0,0}, 2*Pi} { Line{5} ; Layers{1}; Recombine;};
Coherence; // remove duplicate lines

/*
 * Surface Diagram 
 *    ______    _____
 *   |      \__/     |
 *  3|  2    _1   4  |5
 *   |______/  \_____|
 *
 */
/*
Surface Loop(20) = {1};
Surface Loop(21) = {2};
Surface Loop(22) = {3};
Surface Loop(23) = {4};
Surface Loop(24) = {5};
Volume(100)      = {20,21,22,23,24};
*/
Surface Loop(20) = {1,2,3,4,5};
Volume(100)      = {20};

Physical Surface(2)  = {1, 2, 4};
Physical Surface(0)  = {3}; 
Physical Surface(1)  = {5}; 
Physical Volume(101) = {100};
Mesh.Smoothing = 10;
