close all
clear all
clc

%colorhex = {'ff7f0e','d62728', '9467bd','8c564b','2ca02c','1f77b4'};

% Use ligher colors for better readability
colorhex = {'ffbf86','ea9393', 'c9b3de','c5aaa5','95cf95','8fbbd9'};
for cc = 1 : size(colorhex,2)
    color(cc,:) = sscanf(colorhex{cc},'%2x%2x%2x',[1 3])/255;
end


figure
for horlayers = [0 30 60]
    azi = [0:360];
    ele = horlayers*1.05*ones(size(azi));
    [x,y,z] = sph2cart(azi*pi/180,ele*pi/180,ones(size(azi)));
    plot3(x,y,z,'--','Color',0.5*[1 1 1])
    hold on
end

% for vertlayers = [0 -30 -90]
%     if vertlayers == 0
%         ele = [-15:30];
%     else
%         ele = [0:30];
%     end
%     azi = vertlayers*ones(size(ele))
%     [x,y,z] = sph2cart(azi*pi/180,ele*pi/180,ones(size(azi)));
%     plot3(x,y,z,':','Color',0.5*[1 1 1])
% end

for vertlayers = [0 -30 -90 30 90 150 -150 180]
    if vertlayers == 0
        ele = [-15:90];
    else
        ele = [0:90];
    end
    azi = vertlayers*ones(size(ele))
    [x,y,z] = sph2cart(azi*pi/180,ele*pi/180,ones(size(azi)));
    plot3(x,y,z,':','Color',0.5*[1 1 1])
end



azi = [0 30 90 150 180 -150 -90 -30 0 30 90 150 180 -150 -90 -30 0 90 180 -90 0 -90 -30 0 0]
ele = [0 0 0 0 0 0 0 0 30 30 30 30 30 30 30 30 60 60 60 60 90 15 15 15 -15]
[x,y,z] = sph2cart(azi*pi/180,ele*pi/180,1.03*ones(size(azi)));
col = [1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 4 5 5 5 6]


for ii = 1 : size(x,2)
    if abs(azi(ii))<90
        alpha = 1;
    else
        alpha = 0.7;
    end
    scatter3(x(ii),y(ii),z(ii),100+100*(180-abs(azi(ii)))/180,'o','MarkerEdgeColor',0.3*[1 1 1],'MarkerFaceColor',color(col(ii),:),'markerfacealpha',alpha)
    (180-abs(azi(ii)))/180
    text(x(ii),y(ii),z(ii),num2str(ii),'HorizontalAlignment','center','VerticalAlignment','middle')
end
view(100,4)

fill3([-1 1 1 -1],[-1 -1 1 1],[-0.5 -0.5 -0.5 -0.5],'w') 


[X,Y,Z] = sphere(100);

surf(0.1*X,0.1*Y,0.1*Z,'EdgeColor','none','FaceColor',0.5*[1 1 1],'FaceAlpha',1)
%surf(0.2*X,0.2*Y,0.2*Z-0.3,'EdgeColor','none','FaceColor',0.5*[1 1 1],'FaceAlpha',1)

surf(X,Y,abs(Z),'EdgeColor','none','FaceColor',0.8*[1 1 1],'FaceAlpha',0.2)
% fill3(X(:),Y(:),Z(:),0.8*[1 1 1],'EdgeColor','none','FaceAlpha',0.1)


plot3([0 0.11 0],[0.11 0 -0.11],[0 0 0],'k','LineWidth',4)
axis equal
axis off
print('-dpng','-r600','layout3d.png')