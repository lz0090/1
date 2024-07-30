load('ever_delay.mat')
load('evert_t.mat')
load('random.mat')
load('URM_t.mat')
hold on
grid on
box on
set(gca,'GridLineStyle','--','GridColor','k','GridAlpha',1); % ':'：网格线虚线；'-'：网格线实线
set(gca,'FontSize',17);
set(gca,'FontName','Times New Roman');
plot(unnamed ,'color',addcolor(108),'linewidth', 1.8);
plot(URM_T ,'color',addcolor(162), 'linewidth', 1.8);
plot(ever_delay ,'color',addcolor(23), 'linewidth', 1.8);
plot(unnamed1,'color',addcolor(121), 'linewidth', 1.8);
% plot(27, 58.791, 'bo', 'MarkerSize', 9);
% plot(27, 48.967, 'ro', 'MarkerSize', 9);
xlabel('Slot'); 
ylabel('Average delay of each slot (ms)');  
axis([1,50,35,60])
legend('RRM','URM','P-DRR with \psi','P-DRR without \psi'); 
hold off