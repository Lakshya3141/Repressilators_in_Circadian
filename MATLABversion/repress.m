function[derivatives]= repress(t,D,v)
%calculating derivatives for protein and mrna levels
ma=D(1);
pa=D(2);
mb=D(3);
pb=D(4);
mc=D(5);
pc=D(6);
% a1=para(1,1);
% a2=para(2,1);
% a3=para(3,1);
% b1=para(4,1);
% b2=para(5,1);
% b3=para(6,1);
% n=para(7,1);
n=2;
derivatives = [-ma+ (v(1)/(1+pc^n));
-v(4)*(pa - ma);
-mb+ (v(2)/(1+pa^n));
-v(5)*(pb - mb);
-mc+ (v(3)/(1+pb^n));
-v(6)*(pc - mc)];
end