set(0,'defaultTextInterpreter','latex');

% Domain Size
Lx = 1;
Ly = 1;
Lz = 1;

% Grid Size
res = 128;
Nx = res*Lx;
Ny = res*Ly;
Nz = res*Lz;

% Grid
x = permute((0:Nx-1) ,[2 1 3])/Nx*Lx;
y = permute((0:Ny-1) ,[1 2 3])/Ny*Ly;
z = permute((0:Nz-1) ,[3 1 2])/Nz*Lz;

% waves in x-direction & y-direction & z-direction
kn = complex(zeros(Nx,Ny,Nz,3,'gpuArray'));
kn(:,:,:,1) = 2*pi/Lx*permute( fftshift(-Nx/2:Nx/2-1) ,[2 1 3]) + y*0+z*0;
kn(:,:,:,2) = 2*pi/Ly*permute( fftshift(-Ny/2:Ny/2-1) ,[1 2 3]) + z*0+x*0;
kn(:,:,:,3) = 2*pi/Lz*permute( fftshift(-Nz/2:Nz/2-1) ,[3 1 2]) + x*0+y*0;
lap = sum((1i*kn).^2,4); % Laplace

% Spacing
dx = x(2)-x(1);
dy = y(2)-y(1);
dz = z(2)-z(1);

% Time parameter
dt = .1*dx; % initial time step
t = gather(0); % time step

% Kinematic Viscosity
nu = .001;

% external force
f_hat = zeros(Nx,Ny,Nz,3,'gpuArray');
f_hat(:,:,:,1) = fftn(20*(exp(-500*((x-.5).^2+(y-.5).^2+(z-.5).^2))), [Nx Ny Nz]);
f_hat(:,:,:,2) = f_hat(:,:,:,1);
f_hat(:,:,:,3) = f_hat(:,:,:,1);

u = zeros(Nx,Ny,Nz,3,'gpuArray');
u_hat = zeros(Nx,Ny,Nz,3,'gpuArray');
p_hat = zeros(Nx,Ny,Nz,1,'gpuArray');
convect = zeros(Nx,Ny,Nz,3,'gpuArray');


time = linspace(0.1,100,3000);
dir = zeros(1,1,1,3);
clf;p = patch;
v = VideoWriter('newfile2.avi','Motion JPEG AVI');
v.Quality = 80;
v.FrameRate = 24;
open(v);
% dw/dt = - u*dw/dx - v*dw/dy + nu*(d^2w/dx^2 + d^2w/dy^2) + f
for k = 1:length(time)
    while t < time(k)
        dir(1) = sin(gather(t))*cos(gather(2*t)); dir(2) = cos(gather(t))*cos(gather(2*t)); dir(3) = sin(gather(2*t));
        
        % Convection Terms
        % {\partial}(u_i*u_j)/{\partial} x_j 
        for i = 1:3
            u(:,:,:,i) = ifftn(u_hat(:,:,:,i),[Nx Ny Nz]);
        end
        for i = 1:3
            convect(:,:,:,i) = 0;
            for j = 1:3
                convect(:,:,:,i) = convect(:,:,:,i) + 1i*kn(:,:,:,j).*fftn(u(:,:,:,j).*u(:,:,:,i),[Nx Ny Nz]);
            end
        end
        % Compute dt
        dt = min(0.5*dx/max(abs(u(:))),time(k)-t);
        
        % Explicit Euler for convection and Modified Implicit Euler for Diffusion
        u_hat = (u_hat.*(1/dt + nu*lap) + f_hat.*dir - convect)./(1/dt - nu*lap);
        
        % laplacian(p) = 1/dt*divergence(u)
        p_hat = sum(1i*kn.*u_hat,4)./lap; p_hat(1) = 0; %p_hat = zeros(M,N);
        
        % Correct Velocities
        u_hat = u_hat - 1i*kn.*p_hat;
        
        % time update
        t = t + dt;
        
        % time print
        fprintf('t = %.3f\n',t);
    end
    cd = [3 2;1 3;2 1];
    vort = zeros(Nx,Ny,Nz,'gpuArray');
    for i = 1:3
        vort = vort + ifftn(1i*kn(:,:,:,cd(i,2)).*u_hat(:,:,:,cd(i,1)) -...
            1i*kn(:,:,:,cd(i,1)).*u_hat(:,:,:,cd(i,2)),[Nx Ny Nz]).^2;
    end
    vort = sqrt(vort);
    views = {'Front View','Side View','Top View'};
    for i = [1 3]
        subplot(2,3,i+(i>2))
        plotvar = permute(gather(sum(abs(vort),i)),[setdiff(1:3,i) i]);
        imagesc(x,y,plotvar')
        colormap(gca,'bone')
        title(['Vorticit Magnitude, ' views{i}])
        caxis([0 500])
        axis square
        set(gca,'TickLabelInterpreter', 'latex');
    end
    drawnow;
    
    % plotvar = gather(real(u));
    
    %     clf;
    %     vp = 3;
    %     quiver3(x(1:vp:end)+(y(1:vp:end)+z(1:vp:end))*0,y(1:vp:end)+(x(1:vp:end)+z(1:vp:end))*0,...
    %         z(1:vp:end)+(x(1:vp:end)+y(1:vp:end))*0,plotvar((1:vp:end),(1:vp:end),(1:vp:end),1),...
    %         plotvar((1:vp:end),(1:vp:end),(1:vp:end),2),plotvar((1:vp:end),...
    %         (1:vp:end),(1:vp:end),3),'ShowArrowHead','off')
    subplot(2,3,[2 3 5 6])
    delete(p)
%     plotvar = gather(abs(sum(u(:,:,:,:).^2,4)));
    plotvar = vort;
    p = patch(isosurface(x+(y+z)*0,y+(x+z)*0,z+(x+y)*0,plotvar,5));
    phi = griddedInterpolant(x+(y+z)*0,y+(x+z)*0,z+(x+y)*0,gather(real(ifftn(p_hat,[Nx Ny Nz]))));
    p.FaceVertexCData = 1/gather(dt)*phi(p.Vertices(:,1),p.Vertices(:,2),p.Vertices(:,3));

    p.FaceColor = 'interp';
    p.EdgeColor = 'k';
    p.EdgeAlpha = .1;
    grid on; grid minor;
    view(3);
    colormap(gca,'jet')
    cb = colorbar;
    ylabel(cb, 'Pressure')
%     cb.Label.Interpreter = 'latex';
    xlabel('x-axis (m)'); ylabel('y-axis (m)'); zlabel('z-axis (m)')
    title([sprintf('Iso-Surface of $|\\nabla{\\times \\vec{v}}| = 5$ with $\\nu = 0.001 $ at $t = %.3f$',gather(t)), char(10),...
        '$|\vec{f}| = 20 e^{-500((x-0.5)^2+(y-0.5)^2+(z-0.5)^2)}$ ',...
        'Force Direction: $\theta = ' num2str(gather(t),'%.1f') ', \phi = ' num2str(gather(2*t),'%.1f') '$'],...
        'fontsize',15,'interpreter','latex','fontsize',15)
    caxis([-.16 .06])
    axis tight
    set(gca,'TickLabelInterpreter', 'latex');
% 20*(exp(-500*((x-.5).^2+(y-.5).^2+(z-.5).^2)))
    axis([0 Lx 0 Ly 0 Lz]); 
    writeVideo(v,getframe(gcf));


end