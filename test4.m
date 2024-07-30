filepath = 'E:\project\PPO_static\test3mat\';

load([filepath 'evalu_compre' num2str(10) '.mat']);
load([filepath 'evalu_compre' num2str(30) '.mat']);
load([filepath 'evalu_compre' num2str(50) '.mat']);
load([filepath 'evalu_compre' num2str(70) '.mat']);
load([filepath 'evalu_compre' num2str(90) '.mat']);
hold on
grid


plot(evalu_compre10,'g', 'linewidth', 1.6);
plot(evalu_compre30,'r', 'linewidth', 1.6);
plot(evalu_compre50,'m', 'linewidth', 1.6);
plot(evalu_compre70,'b', 'linewidth', 1.6);
plot(evalu_compre90,'c', 'linewidth', 1.6);
xlabel('Threshold'); 
ylabel('Comprehensive evaluation'); 
set(gca,'FontSize',11);
set(gca,'FontName','Times New Roman');
legend('\alpha = 0.1','\alpha = 0.3','\alpha = 0.5','\alpha = 0.7','\alpha = 0.9')
hold off