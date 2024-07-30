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

for i = 1:5
    b = bar(i:i+1,[num_de10(i,:);0,0],1,'stacked');
    set(b(1),'facecolor',color_matrix(i,:) )
    set(b(2),'facecolor',color_matrix(i,:))
    if i == 1
        b1 = b(1);
    end
    if i == 2
        b2 = b(1);
    end
    if i == 3
        b3 = b(1);
    end
    if i == 4
        b4 = b(1);
    end
    if i == 5
        b5 = b(1);
    end
    a1 = hatchfill2(b(1),'single','HatchAngle',0,'HatchDensity',90,'HatchColor','k');
    hatchfill2(b(2),'single','HatchAngle',45,'HatchDensity',60,'HatchColor','k');
end

% 取底色便于绘制填充图例
for i = 1:5
    b = bar(i+6:i+7,[num_de30(i,:);0,0],1,'stacked');
    G0 = b
    set(b(1),'facecolor',addcolor(9) )
    set(b(2),'facecolor',addcolor(9))
%     hatchfill2(b(1),'single','HatchAngle',0,'HatchDensity',90,'HatchColor','k');
%     hatchfill2(b(2),'single','HatchAngle',45,'HatchDensity',60,'HatchColor','k');
end

for i = 1:5
    b = bar(i+6:i+7,[num_de30(i,:);0,0],1,'stacked');
    set(b(1),'facecolor',color_matrix(i,:) )
    set(b(2),'facecolor',color_matrix(i,:))
    hatchfill2(b(1),'single','HatchAngle',0,'HatchDensity',90,'HatchColor','k');
    hatchfill2(b(2),'single','HatchAngle',45,'HatchDensity',60,'HatchColor','k');
end



for i = 1:5
    b = bar(i+12:i+13,[num_de50(i,:);0,0],1,'stacked');
    set(b(1),'facecolor',color_matrix(i,:) )
    set(b(2),'facecolor',color_matrix(i,:))
    hatchfill2(b(1),'single','HatchAngle',0,'HatchDensity',90,'HatchColor','k');
    hatchfill2(b(2),'single','HatchAngle',45,'HatchDensity',60,'HatchColor','k');
end

for i = 1:5
    b = bar(i+18:i+19,[num_de70(i,:);0,0],1,'stacked');
    set(b(1),'facecolor',color_matrix(i,:) )
    set(b(2),'facecolor',color_matrix(i,:))
    hatchfill2(b(1),'single','HatchAngle',0,'HatchDensity',90,'HatchColor','k');
    hatchfill2(b(2),'single','HatchAngle',45,'HatchDensity',60,'HatchColor','k');
end

for i = 1:5
    b = bar(i+24:i+25,[num_de90(i,:);0,0],1,'stacked');
    set(b(1),'facecolor',color_matrix(i,:) )
    set(b(2),'facecolor',color_matrix(i,:))
    hatchfill2(b(1),'single','HatchAngle',0,'HatchDensity',90,'HatchColor','k');
    hatchfill2(b(2),'single','HatchAngle',45,'HatchDensity',60,'HatchColor','k');
end


set(gca, 'Box', 'off', ...                                         % 边框
         'XGrid', 'off', 'YGrid', 'on', ...                        % 网格
         'TickDir', 'out', 'TickLength', [.02 .02], ...            % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1],...           % 坐标轴颜色
         'YTick', 0:5:40,...                                      % 刻度位置、间隔
         'XTick', 3:6:27,...                                      % 刻度位置、间隔
         'Ylim' , [0 40], ...                                     % 坐标轴范围
         'Xticklabel',{ '0.05' '0.15' '0.25' '0.35'  '0.45' },...% X坐标轴刻度标签
         'Yticklabel',{[0:5:40]})                                 % Y坐标轴刻度标签

hYLabel = ylabel('Number of reallocation');
hXLabel = xlabel('Threshold');
set([hYLabel,hXLabel], 'FontName',  'Times New Roman')
set([hYLabel,hXLabel], 'FontSize', 17)




%% 时延曲线
yyaxis right
x_delay = [3 9 15 21 27]
ylim([40 47])
% 刻度标签字体和字号
set(gca, 'FontName', 'Times New Roman', 'FontSize', 17)
ylabel('Average delay of all slots(ms)','FontName','Times New Roman','FontSize',17)
set(gca,'ycolor',[.1 .1 .1]);
d1 = plot(x_delay,delay10,'-.','color',color_matrix(1,:), 'linewidth', 1.6, 'marker', 's');
d2 = plot(x_delay,delay30,'-.','color',color_matrix(2,:), 'linewidth', 1.6, 'marker', 'd');
d3 = plot(x_delay,delay50,'-.','color',color_matrix(3,:), 'linewidth', 1.6, 'marker', 'p');
d4 = plot(x_delay,delay70,'-.','color',color_matrix(4,:), 'linewidth', 1.6, 'marker', '^');
d5 = plot(x_delay,delay90,'-.','color',color_matrix(5,:), 'linewidth', 1.6, 'marker', 'x');


%% 图例显示

% 绘制填充图例

% legendData = { '\alpha = 0.1', '\alpha = 0.3', '\alpha = 0.5','\alpha = 0.7','\alpha = 0.9','A-reallocation','P-reallocation'};
% [legend_h, object_h, plot_h, text_str] = legendflex([b1(1),b2(1),b3(1),b4(1),b5(1),G0(1),G0(2)], legendData, 'Padding', [2, 1, 10], 'FontSize', 10,'FontName',  'Times New Roman');
% hatchfill2(object_h(13), 'single', 'HatchAngle', 0, 'HatchDensity', 90, 'HatchColor', 'k');
% hatchfill2(object_h(14), 'single', 'HatchAngle', 45, 'HatchDensity', 60, 'HatchColor', 'k');

legendData = {'A-reallocation','P-reallocation'};
[legend_h, object_h, plot_h, text_str] = legendflex([G0(1),G0(2)], legendData, 'Padding', [2, 1, 1], 'FontSize', 10,'FontName',  'Times New Roman');
hatchfill2(object_h(3), 'single', 'HatchAngle', 0, 'HatchDensity', 20, 'HatchColor', 'k');
hatchfill2(object_h(4), 'single', 'HatchAngle', 45, 'HatchDensity', 15, 'HatchColor', 'k');

hLegend = legend([b1(1),b2(1),b3(1),b4(1),b5(1)], ...
    '\alpha = 0.1', '\alpha = 0.3', '\alpha = 0.5','\alpha = 0.7','\alpha = 0.9');
% 标签及Legend的字体字号 
set([hLegend], 'FontName',  'Times New Roman')
set([hLegend], 'FontSize', 10)

% 刻度标签字体和字号
set(gca, 'FontName', 'Times New Roman', 'FontSize', 17)


% 背景颜色
set(gca,'Color',[1 1 1])

















