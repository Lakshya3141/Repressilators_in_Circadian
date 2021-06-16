clear all; close all; clc;
time = linspace(0,1000,100);
rng(1998);
IC = [rand,rand,rand,rand,rand,rand];
para = [800;1000;1000;6;7;10];
[t,y] = ode45(@(t,y) repress(t, y, para),time,IC);
ma= y(:,1);
pa= y(:,2);
mb= y(:,3);
pb= y(:,4);
mc= y(:,5);
pc= y(:,6);
plot(t,pa,'LineWidth',1.5); hold on
plot(t,pb,'LineWidth',1.5); hold on
plot(t,pc,'LineWidth',1.5);
ylabel("Proteins per cell in aribitrary units");
xlabel("time in minutes");
legend("Cry1","Per2","Reverb-a");