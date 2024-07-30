filepath = 'E:\project\PPO_static\mat\delay\';
n = 50;

for i=1:n
   load([filepath 'delay' num2str(i) '.mat']);
end
hold on
grid

% plot(delay1,'g', 'linewidth', 1.6);
% plot(delay2,'r', 'linewidth', 1.6);
% plot(delay3,'m', 'linewidth', 1.6);
% plot(delay4,'b', 'linewidth', 1.6);
% plot(delay5,'c', 'linewidth', 1.6);
% xlabel('\alpha'); 
% ylabel('Average delay of all slots (ms)'); 
% set(gca,'FontSize',11);
% set(gca,'FontName','Times New Roman');
% legend('Threshold = 0.01','Threshold = 0.02','Threshold = 0.03','Threshold = 0.04','Threshold = 0.05')

% plot(delay6,'g', 'linewidth', 1.6);
% plot(delay7,'r', 'linewidth', 1.6);
% plot(delay8,'m', 'linewidth', 1.6);
% plot(delay9,'b', 'linewidth', 1.6);
% plot(delay10,'c', 'linewidth', 1.6);
% xlabel('\alpha'); 
% ylabel('Average delay of all slots (ms)'); 
% set(gca,'FontSize',11);
% set(gca,'FontName','Times New Roman');
% legend('Threshold = 0.06','Threshold = 0.07','Threshold = 0.08','Threshold = 0.09','Threshold = 0.10')

% plot(delay11,'g', 'linewidth', 1.6);
% plot(delay12,'r', 'linewidth', 1.6);
% plot(delay13,'m', 'linewidth', 1.6);
% plot(delay14,'b', 'linewidth', 1.6);
% plot(delay15,'c', 'linewidth', 1.6);
% xlabel('\alpha'); 
% ylabel('Average delay of all slots (ms)'); 
% set(gca,'FontSize',11);
% set(gca,'FontName','Times New Roman');
% legend('Threshold = 0.11','Threshold = 0.12','Threshold = 0.13','Threshold = 0.14','Threshold = 0.15')

% plot(delay16,'g', 'linewidth', 1.6);
% plot(delay17,'r', 'linewidth', 1.6);
% plot(delay18,'m', 'linewidth', 1.6);
% plot(delay19,'b', 'linewidth', 1.6);
% plot(delay20,'c', 'linewidth', 1.6);
% xlabel('\alpha'); 
% ylabel('Average delay of all slots (ms)'); 
% set(gca,'FontSize',11);
% set(gca,'FontName','Times New Roman');
% legend('Threshold = 0.16','Threshold = 0.17','Threshold = 0.18','Threshold = 0.19','Threshold = 0.20')

% plot(delay21,'g', 'linewidth', 1.6);
% plot(delay22,'r', 'linewidth', 1.6);
% plot(delay23,'m', 'linewidth', 1.6);
% plot(delay24,'b', 'linewidth', 1.6);
% plot(delay25,'c', 'linewidth', 1.6);
% xlabel('\alpha'); 
% ylabel('Average delay of all slots (ms)'); 
% set(gca,'FontSize',11);
% set(gca,'FontName','Times New Roman');
% legend('Threshold = 0.21','Threshold = 0.22','Threshold = 0.23','Threshold = 0.24','Threshold = 0.25')


% plot(delay26,'g', 'linewidth', 1.6);
% plot(delay27,'r', 'linewidth', 1.6);
% plot(delay28,'m', 'linewidth', 1.6);
% plot(delay29,'b', 'linewidth', 1.6);
% plot(delay30,'c', 'linewidth', 1.6);
% xlabel('\alpha'); 
% ylabel('Average delay of all slots (ms)'); 
% set(gca,'FontSize',11);
% set(gca,'FontName','Times New Roman');
% legend('Threshold = 0.26','Threshold = 0.27','Threshold = 0.28','Threshold = 0.29','Threshold = 0.30')





plot(delay46,'g', 'linewidth', 1.6);
plot(delay47,'r', 'linewidth', 1.6);
plot(delay48,'m', 'linewidth', 1.6);
plot(delay49,'b', 'linewidth', 1.6);
plot(delay50,'c', 'linewidth', 1.6);
xlabel('\alpha'); 
ylabel('Average delay of all slots (ms)'); 
set(gca,'FontSize',11);
set(gca,'FontName','Times New Roman');
legend('Threshold = 0.46','Threshold = 0.47','Threshold = 0.48','Threshold = 0.49','Threshold = 0.50')
