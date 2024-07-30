%% 数据准备
x = [0.05 0.15 0.25 0.35 0.45];
num_de10=[36 0; 12 8; 7 6; 5 6; 2 6];
num_de30=[29 1; 10 8; 6 8; 3 9; 1 3];
num_de50=[25 2; 8 4; 4 4; 2 3; 2 4];
num_de70=[20 7; 8 10; 4 7; 3 7; 2 6];
num_de90=[24 5; 10 9; 6 7; 4 12; 3 10];
Y=[num_de10;num_de30;num_de50;num_de70;num_de90];
num_all10=[36 20 13 11 8];
num_all30=[30 18 14 12 4];
num_all50=[27 12 8 5 6];
num_all70=[27 18 11 10 8];
num_all90=[29 19 13 16 13];
delay10=[43.9625 44.2855 44.3385 44.6285 45.5853];
delay30=[44.1758 44.3384 44.4191 45.5251 46.0555];
delay50=[44.4313 45.2480 45.8331 45.9069 45.9931];
delay70=[44.3709 45.0028 45.4416 45.6737 45.8331];
delay90=[43.9907 44.3672 44.6621 44.9388 45.3461];
dataset = [num_all10;num_all30;num_all50;num_all70;num_all90]';
%% 颜色定义
% addcolor函数中270种颜色对照表（见文章底部）
% https://blog.csdn.net/qq_37233260/article/details/118642983
C1 = addcolor(166); 
C2 = addcolor(107);
C3 = addcolor(178);
C4 = addcolor(119);% 紫色
C5 = addcolor(100)

%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 20;
figureHeight = 15;

%% 柱状图绘制
%窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]); % define the new figure dimensions
hold on

% 绘制柱图
% 1-调节柱间距
GO = bar(dataset,1,'EdgeColor','k');








% 赋色
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
GO(3).FaceColor = C3;
GO(4).FaceColor = C4;
GO(5).FaceColor = C5;

% 文字注释，不需要可删
% for ii=1:5
%     text(ii-0.24,dataset(ii,1)+0.005,num2str(dataset(ii,1)),...
%          'ROtation',90,'color',C1,'FontSize',10,'FontName',  'Helvetica');
%     text(ii,dataset(ii,2)+0.01,num2str(dataset(ii,2)),...
%          'ROtation',90,'color',C2,'FontSize',10,'FontName',  'Helvetica');     
%     text(ii+0.22,dataset(ii,3)+0.01,num2str(dataset(ii,3)),...
%          'ROtation',90,'color',C3,'FontSize',10,'FontName',  'Helvetica');  
% end

% 坐标区调整
set(gca, 'Box', 'off', ...                                         % 边框
         'XGrid', 'off', 'YGrid', 'on', ...                        % 网格
         'TickDir', 'out', 'TickLength', [.02 .02], ...            % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1],...           % 坐标轴颜色
         'YTick', 0:10:70,...                                      % 刻度位置、间隔
         'Ylim' , [0 70], ...                                     % 坐标轴范围
         'Xticklabel',{ '' '0.05' '' '0.15' '' '0.25' '' '0.35' '' '0.45' },...% X坐标轴刻度标签
         'Yticklabel',{[0:10:70]})                                 % Y坐标轴刻度标签

% 标签及Legend 设置    
hYLabel = ylabel('Number of reallocation');
hXLabel = xlabel('Threshold');
hLegend = legend([GO(1),GO(2),GO(3),GO(4),GO(5)], ...
    '\alpha = 0.1', '\alpha = 0.3', '\alpha = 0.5','\alpha = 0.7','\alpha = 0.9');
% Legend位置微调 
P = hLegend.Position;
hLegend.Position = P + [0.015 0.03 0 0];

% 刻度标签字体和字号
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)
% 标签及Legend的字体字号 
set([hYLabel,hXLabel,hLegend], 'FontName',  'Times New Roman')
set([hYLabel,hXLabel,hLegend], 'FontSize', 15)

% 背景颜色
set(gca,'Color',[1 1 1])

%% 1
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'eg';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');


%% 时延曲线
yyaxis right
ylim([40 47])
ylabel('Average delay of all slots(ms)','FontName','Times New Roman','FontSize',15)
set(gca,'ycolor',[.1 .1 .1]);
plot(delay10,'-.','color',C1, 'linewidth', 1.6, 'marker', 's');
plot(delay30,'-.','color',C2, 'linewidth', 1.6, 'marker', 'd');
plot(delay50,'-.','color',C3, 'linewidth', 1.6, 'marker', 'p');
plot(delay70,'-.','color',C4, 'linewidth', 1.6, 'marker', '^');
plot(delay90,'-.','color',C5, 'linewidth', 1.6, 'marker', 'x');


