clc;
clear;

%% 数据准备
color_matrix = [addcolor(108)                  %1号柱下颜色
                addcolor(107)                  %1号柱上颜色
                addcolor(166)                  %2号柱下颜色
                addcolor(165)                  %2号柱上颜色
                addcolor(121)                  %3号柱下颜色
                addcolor(119)                  %3号柱上颜色
                addcolor(242)                  %4号柱下颜色
                addcolor(243)                  %4号柱上颜色
                addcolor(183)                  %5号柱下颜色
                addcolor(184)                  %5号柱上颜色
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

%% 柱状图绘制

for i = 1:30
    if mod(i,6) == 1 
        b = bar(i,Y(i),1,'FaceColor',addcolor(11) ,'EdgeColor',color_matrix(1,:));
    elseif mod(i,6) == 2 
        b = bar(i,Y(i),1,'FaceColor',addcolor(9) ,'EdgeColor',color_matrix(1,:));  
    elseif mod(i,6) == 3 
        b = bar(i,Y(i),1,'FaceColor',addcolor(13) ,'EdgeColor',color_matrix(1,:)); 
    elseif mod(i,6) == 4 
        b = bar(i,Y(i),1,'FaceColor',addcolor(14) ,'EdgeColor',color_matrix(1,:)); 
    elseif mod(i,6) == 5 
        b = bar(i,Y(i),1,'FaceColor',addcolor(15) ,'EdgeColor',color_matrix(1,:));
    elseif mod(i,6) == 0 
        b = bar(i,Y(i),1,'FaceColor',addcolor(15) ,'EdgeColor',color_matrix(1,:));
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  1
b = bar(1:2,[num_de10(1,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(108)  )
set(b(2),'facecolor',addcolor(107) )
b = bar(2:3,[num_de10(2,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(183)  )
set(b(2),'facecolor',addcolor(165) )
b = bar(3:4,[num_de10(3,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(121)  )
set(b(2),'facecolor',addcolor(119) )
b = bar(4:5,[num_de10(4,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(162)  )
set(b(2),'facecolor',addcolor(161) )
b = bar(5:6,[num_de10(5,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(23)  )
set(b(2),'facecolor',addcolor(22) )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  2
b = bar(7:8,[num_de30(1,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(108)  )
set(b(2),'facecolor',addcolor(107) )
b = bar(8:9,[num_de30(2,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(183)  )
set(b(2),'facecolor',addcolor(165) )
b = bar(9:10,[num_de30(3,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(121)  )
set(b(2),'facecolor',addcolor(119) )
b = bar(10:11,[num_de30(4,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(162)  )
set(b(2),'facecolor',addcolor(161) )
b = bar(11:12,[num_de30(5,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(23)  )
set(b(2),'facecolor',addcolor(22) )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 3
b = bar(13:14,[num_de50(1,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(108)  )
set(b(2),'facecolor',addcolor(107) )
b = bar(14:15,[num_de50(2,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(183)  )
set(b(2),'facecolor',addcolor(165) )
b = bar(15:16,[num_de50(3,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(121)  )
set(b(2),'facecolor',addcolor(119) )
b = bar(16:17,[num_de50(4,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(162)  )
set(b(2),'facecolor',addcolor(161) )
b = bar(17:18,[num_de50(5,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(23)  )
set(b(2),'facecolor',addcolor(22) )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 4
b = bar(19:20,[num_de70(1,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(108)  )
set(b(2),'facecolor',addcolor(107) )
b = bar(20:21,[num_de70(2,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(183)  )
set(b(2),'facecolor',addcolor(165) )
b = bar(21:22,[num_de70(3,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(121)  )
set(b(2),'facecolor',addcolor(119) )
b = bar(22:23,[num_de70(4,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(162)  )
set(b(2),'facecolor',addcolor(161) )
b = bar(23:24,[num_de70(5,:);0,0],1,'stacked');
set(b(1),'facecolor',addcolor(23)  )
set(b(2),'facecolor',addcolor(22) )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 5
b1 = bar(25:26,[num_de90(1,:);0,0],1,'stacked');
set(b1(1),'facecolor',addcolor(108)  )
set(b1(2),'facecolor',addcolor(107) )
b2 = bar(26:27,[num_de90(2,:);0,0],1,'stacked');
set(b2(1),'facecolor',addcolor(183)  )
set(b2(2),'facecolor',addcolor(165) )
b3 = bar(27:28,[num_de90(3,:);0,0],1,'stacked');
set(b3(1),'facecolor',addcolor(121)  )
set(b3(2),'facecolor',addcolor(119) )
b4 = bar(28:29,[num_de90(4,:);0,0],1,'stacked');
set(b4(1),'facecolor',addcolor(162)  )
set(b4(2),'facecolor',addcolor(161) )
b5 = bar(29:30,[num_de90(5,:);0,0],1,'stacked');
set(b5(1),'facecolor',addcolor(23)  )
set(b5(2),'facecolor',addcolor(22) )






set(gca, 'Box', 'on', ...                                         % 边框
         'XGrid', 'off', 'YGrid', 'on', ...                        % 网格
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1],...           % 坐标轴颜色
         'YTick', 0:5:40,...                                      % 刻度位置、间隔
         'XTick', 3:6:27,...                                      % 刻度位置、间隔
         'Ylim' , [0 40], ...                                     % 坐标轴范围
         'Xticklabel',{ '0.05' '0.15' '0.25' '0.35'  '0.45' },...% X坐标轴刻度标签
         'Yticklabel',{[0:5:40]})                                 % Y坐标轴刻度标签

hYLabel = ylabel('Number of reallocation');
hXLabel = xlabel('\alpha_0');
set([hYLabel,hXLabel], 'FontName',  'Times New Roman')
set([hYLabel,hXLabel], 'FontSize', 15)




% %% 时延曲线
% yyaxis right
% x_delay = [3 9 15 21 27]
% ylim([40 47])
% % 刻度标签字体和字号
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 17)
% ylabel('Average delay of all slots(ms)','FontName','Times New Roman','FontSize',17)
% set(gca,'ycolor',[.1 .1 .1]);
% d1 = plot(x_delay,delay10,'-.','color',color_matrix(1,:), 'linewidth', 1.6, 'marker', 's');
% d2 = plot(x_delay,delay30,'-.','color',color_matrix(2,:), 'linewidth', 1.6, 'marker', 'd');
% d3 = plot(x_delay,delay50,'-.','color',color_matrix(3,:), 'linewidth', 1.6, 'marker', 'p');
% d4 = plot(x_delay,delay70,'-.','color',color_matrix(4,:), 'linewidth', 1.6, 'marker', '^');
% d5 = plot(x_delay,delay90,'-.','color',color_matrix(5,:), 'linewidth', 1.6, 'marker', 'x');
% 

%% 图例显示

% 标签及Legend 设置    
hYLabel = ylabel('Number of reallocation');
hXLabel = xlabel('{\psi}_0');
hLegend = legend([b1(1),b2(1),b3(1),b4(1),b5(1),b1(2),b2(2),b3(2),b4(2),b5(2)], ...
    '\alpha = 0.1,A', '\alpha = 0.3,A', '\alpha = 0.5,A','\alpha = 0.7,A', '\alpha = 0.9,A','\alpha = 0.1,P', '\alpha = 0.3,P','\alpha = 0.5,P', '\alpha = 0.7,P','\alpha = 0.9,P');
% Legend位置微调 
P = hLegend.Position;
hLegend.Position = P + [0.015 0.03 0 0];

% 刻度标签字体和字号
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15)
% 标签及Legend的字体字号 
set([hYLabel,hXLabel], 'FontName',  'Times New Roman')
set([hYLabel,hXLabel], 'FontSize', 15)
set([hLegend], 'FontName',  'Times New Roman')
set([hLegend], 'FontSize', 10)
% 背景颜色
set(gca,'Color',[1 1 1])

















