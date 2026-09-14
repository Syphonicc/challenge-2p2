function [x1,y1] = getBodyCoordinates(AoAdeg,xC,yC,dx)
% function to output body coordinates for flat plate
AoArad = AoAdeg*pi/180;
a0 = -AoArad;

dxBody = dx;
x0 = 0:dxBody:1;
y0 = 0*x0;
x0r = x0 - xC;
y0r = y0- yC;
x1r = x0r*cos(a0)-y0*sin(a0);
y1r = x0r*sin(a0)+y0*cos(a0);
x1 = x1r+xC;
y1 = y1r+yC;