colormap(jet(1024))

% Domain Size
Lx = 2;
Ly = 1;

% Grid Size
M = 512;
N = 256;

% Grid
x = (0:M-1)'/M*Lx;
y = (0:N-1) /N*Ly;

% waves in x-direction & y-direction
kn = complex(zeros(M,N,2));
kn(:,:,1) = 2*pi/Lx*fftshift(-M/2:M/2-1)'+y*0;
kn(:,:,2) = 2*pi/Ly*fftshift(-N/2:N/2-1)+x*0;

lap = sum((1i*kn).^2,3); % Laplace

% Spacing
dx = x(2)-x(1);
dy = y(2)-y(1);

% Time parameter
dt = .1*dx; % initial time step
t = 0; % time step

% Kinematic Viscosity
nu = .001;

% external force
f_x = 20*(exp(-1000*((x-1).^2+(y-.5).^2)))-0*(exp(-1000*((x-1.5).^2+(y-.5).^2)));


u_hat = zeros(M,N,2);
f_hat = zeros(M,N,2);
f_hat(:,:,1) = fft2(f_x);
p_hat = zeros(M,N,1);
convect = zeros(M,N,2);


time = linspace(0.1,100,1000);

% dw/dt = - u*dw/dx - v*dw/dy + nu*(d^2w/dx^2 + d^2w/dy^2) + f
for k = 1:length(time)
    while t < time(k)
        dir = zeros(1,1,2);
        dir(1) = atan(1000*(sin(t))); dir(2) = 0.5*atan(1000*sign(cos(t)));
        
        % Convection Terms
        %d(u_i*u_j)/d(x_j)
        u = ifft2(u_hat);
        for i = 1:2
            convect(:,:,i) =  sum(1i*kn.*fft2(u.*u(:,:,i)),3);
        end
        % Compute stable dt
        dt = min(0.5*min(dx,dy)/max(abs(u(:))),time(k)-t);
        
        % Explicit Euler for convection and Modified Implicit Euler for Diffusion
        u_hat = (u_hat.*(1/dt + nu*lap) + f_hat.*dir - convect)./(1/dt - nu*lap);
        
        % laplacian(p) = 1/dt*divergence(u)
        p_hat = sum(1i*kn.*u_hat,3)./lap; p_hat(1) = 0; %p_hat = zeros(M,N);
        
        % Correct Velocities
        u_hat = u_hat - 1i*kn.*p_hat;
        
        % time update
        t = t + dt;
    end
    fprintf('t = %.3f\n',t);
    plotvar = ifft2(1i*kn(:,:,2).*u_hat(:,:,1)-1i*kn(:,:,1).*u_hat(:,:,2));
    imagesc(x,y,real(plotvar)')
    set(gca,'Ydir','normal'); grid minor;
    caxis([-1 1]*30)
    axis equal
    drawnow;
end

