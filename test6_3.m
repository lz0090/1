clc;
clear;

%% 数据准备
color_matrix = [addcolor(108)                  %1号柱下颜色
                addcolor(107)                  %1号柱上颜色
                addcolor(166)                  %2号柱下颜色
                addcolor(165)                  %2号柱上颜色
                addcolor(121)                  %3号柱下颜色
                ];              
num_de10=[36, 0; 12, 8; 7, 6; 5 ,6; 2, 6 ;0,0];
num_de30=[29, 1; 10, 8; 6 ,8; 3 ,9; 1, 3;0,0];
num_de50=[25, 2; 8 ,4; 4, 4; 2, 3; 2, 4;0,0];
num_de70=[20, 7; 8 ,10; 4 ,7; 3 ,7; 2 ,6;0,0];
num_de90=[24 ,5; 10, 9; 6, 7; 4, 12; 3, 10;0,0];

num_all10=[36, 20, 13 ,11, 8,0];
num_all30=[30, 18, 14, 12, 4,0];
num_all50=[27 ,12, 8, 5 ,6,0];
num_all70=[27, 18, 11 ,10,8,0];
num_all90=[29 ,19, 13, 16 ,13,0];
dataset = [num_all10,num_all30,num_all50,num_all70,num_all90];

delay10=[43.9625 44.2855 44.3385 44.6285 45.5853];
delay30=[44.1758 44.3384 44.4191 45.5251 46.0555];
delay50=[44.4313 45.2480 45.8331 45.9069 45.9931];
delay70=[44.3709 45.0028 45.4416 45.6737 45.8331];
delay90=[43.9907 44.3672 44.6621 44.9388 45.3461];
Y=[num_de10;num_de30;num_de50;num_de70;num_de90];
hold on
grid on
set(gca,'GridLineStyle','--','GridColor','k','GridAlpha',1); % ':'：网格线虚线；'-'：网格线实线
%% 折线图绘制





%% 时延曲线
x_delay = [1 2 3 4 5]
ylim([43.5 47])
d1 = plot(x_delay,delay10,'-','color',addcolor(108), 'linewidth', 1.8, 'marker', 's');
d2 = plot(x_delay,delay30,'-','color',addcolor(183), 'linewidth', 1.8, 'marker', 'd');
d3 = plot(x_delay,delay50,'-','color',addcolor(121), 'linewidth', 1.8, 'marker', 'p');
d4 = plot(x_delay,delay70,'-','color',addcolor(162), 'linewidth', 1.8, 'marker', '^');
d5 = plot(x_delay,delay90,'-','color',addcolor(23), 'linewidth', 1.8, 'marker', 'x');


%% 图例显示

% 标签及Legend 设置    
hYLabel = ylabel('Average delay of all slots(ms)');
hXLabel = xlabel('Threshold');
hLegend = legend([d1,d2,d3,d4,d5], ...
    '\alpha = 0.1', '\alpha = 0.3', '\alpha = 0.5','\alpha = 0.7', '\alpha = 0.9');
% Legend位置微调 
P = hLegend.Position;
hLegend.Position = P + [0.015 0.03 0 0];

set(gca, 'Box', 'on', ...                                         % 边框
         'Xticklabel',{ '0.05' '0.15' '0.25' '0.35'  '0.45' })          


% 刻度标签字体和字号
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15)
% 标签及Legend的字体字号 
set([hYLabel,hXLabel], 'FontName',  'Times New Roman')
set([hYLabel,hXLabel], 'FontSize', 15)
set([hLegend], 'FontName',  'Times New Roman')
set([hLegend], 'FontSize', 16)
% 背景颜色
set(gca,'Color',[1 1 1])

















