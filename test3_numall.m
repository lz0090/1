% filepath = 'E:\project\PPO_static\test2mat\num_all\';
% 
% load([filepath 'num_all' num2str(10) '.mat']);
% load([filepath 'num_all' num2str(30) '.mat']);
% load([filepath 'num_all' num2str(50) '.mat']);
% load([filepath 'num_all' num2str(70) '.mat']);
% load([filepath 'num_all' num2str(90) '.mat']);
% hold on
% grid
% 
% plot(num_all10,'g', 'linewidth', 1.6);
% plot(num_all30,'r', 'linewidth', 1.6);
% plot(num_all50,'m', 'linewidth', 1.6);
% plot(num_all70,'b', 'linewidth', 1.6);
% plot(num_all90,'c', 'linewidth', 1.6);
% xlabel('Threshold'); 
% ylabel('num all'); 
% set(gca,'FontSize',11);
% set(gca,'FontName','Times New Roman');
% legend('\alpha = 0.1','\alpha = 0.3','\alpha = 0.5','\alpha = 0.7','\alpha = 0.9')
% hold off

% filepath = 'E:\project\PPO_static\test2mat\num_psi\';
% 
% load([filepath 'num_psi' num2str(10) '.mat']);
% load([filepath 'num_psi' num2str(30) '.mat']);
% load([filepath 'num_psi' num2str(50) '.mat']);
% load([filepath 'num_psi' num2str(70) '.mat']);
% load([filepath 'num_psi' num2str(90) '.mat']);
% hold on
% grid
% 
% plot(num_all10,'g', 'linewidth', 1.6);
% plot(num_all30,'r', 'linewidth', 1.6);
% plot(num_all50,'m', 'linewidth', 1.6);
% plot(num_all70,'b', 'linewidth', 1.6);
% plot(num_all90,'c', 'linewidth', 1.6);
% xlabel('Threshold'); 
% ylabel('num all'); 
% set(gca,'FontSize',11);
% set(gca,'FontName','Times New Roman');
% legend('\alpha = 0.1','\alpha = 0.3','\alpha = 0.5','\alpha = 0.7','\alpha = 0.9')
% hold off

filepath = 'E:\project\PPO_static\test2mat\num_1\';

load([filepath 'num_1' num2str(10) '.mat']);
load([filepath 'num_1' num2str(30) '.mat']);
load([filepath 'num_1' num2str(50) '.mat']);
load([filepath 'num_1' num2str(70) '.mat']);
load([filepath 'num_1' num2str(90) '.mat']);
hold on
grid

plot(num_110,'g', 'linewidth', 1.6);
plot(num_130,'r', 'linewidth', 1.6);
plot(num_150,'m', 'linewidth', 1.6);
plot(num_170,'b', 'linewidth', 1.6);
plot(num_190,'c', 'linewidth', 1.6);
% line([1 5],[0 0],'linestyle','--', 'Color','k', 'LineWidth',1);
xticks([1 2 3 4 5])
xlabel('Threshold'); 
ylabel('Difference'); 
set(gca,'FontSize',16);
set(gca,'FontName','Times New Roman');
legend('\alpha = 0.1','\alpha = 0.3','\alpha = 0.5','\alpha = 0.7','\alpha = 0.9')
hold off