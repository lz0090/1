filepath = 'E:\project\PPO_static\test2mat\delay\';

load([filepath 'delay' num2str(10) '.mat']);
load([filepath 'delay' num2str(30) '.mat']);
load([filepath 'delay' num2str(50) '.mat']);
load([filepath 'delay' num2str(70) '.mat']);
load([filepath 'delay' num2str(90) '.mat']);
hold on
grid

plot(delay10,'g', 'linewidth', 1.6);
plot(delay30,'r', 'linewidth', 1.6);
plot(delay50,'m', 'linewidth', 1.6);
plot(delay70,'b', 'linewidth', 1.6);
plot(delay90,'c', 'linewidth', 1.6);
xlabel('Threshold'); 
xticks([1 2 3 4 5])
ylabel('Average delay of all slots (ms)'); 
set(gca,'FontSize',11);
set(gca,'FontName','Times New Roman');
legend('\alpha = 0.1','\alpha = 0.3','\alpha = 0.5','\alpha = 0.7','\alpha = 0.9')
hold off