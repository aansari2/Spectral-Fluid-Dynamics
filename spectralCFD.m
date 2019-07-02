colormap(jet(1024))

% Domain Size
Lx = 2*pi;
Ly = 2*pi;

% Grid Size
M = 256;
N = 256;

% waves in x-direction & y-direction
kx = fftshift(-M/2:M/2-1)';
ky = fftshift(-N/2:N/2-1)+kx*0;
lap = (1i*kx).^2+(1i*ky).^2; % Laplace

% Grid
x = (0:M-1)'/M*Lx;
y = (0:N-1) /N*Ly;

% Spacing
dx = x(2)-x(1);
dy = y(2)-y(1);

% Time parameter
dt = .1*dx; % initial time step
t = 0; % time step

% Kinematic Viscosity
nu = .01;

% x & y derivative functions
px = @(y) real(ifft2(1i*kx.*fft2(y)));
py = @(y) real(ifft2(1i*ky.*fft2(y)));


% external force
f = 10*(exp(-10*((x-pi).^2+(y-pi).^2)));


u_hat = fft2(x*0+y*0);
v_hat = fft2(x*0+y*0);
w_hat = zeros(M,N);


time = linspace(0,100,1000);

% dw/dt = - u*dw/dx - v*dw/dy + nu*(d^2w/dx^2 + d^2w/dy^2) + f
for k = 1:length(time)
    while t < time(k)
        % Rotate force
        f_hat = fft2(cos(t)*py(f) + sin(t)*px(f));
        
        % Retrieve u and v
        u = ifft2(u_hat);
        v = ifft2(v_hat);
        
        % Convection Terms
        uwx_hat = fft2(u.*ifft2(1i*kx.*w_hat));
        vwy_hat = fft2(v.*ifft2(1i*ky.*w_hat));
        
        % Compute dt
        dt = min(0.5*min(dx,dy)/max(abs([u(:);v(:)])),time(k)-t);
        
        % Explicit Euler for convection and Modified Implicit Euler for Diffusion
        w_hat = (w_hat.*(1/dt + nu*lap) + f_hat - uwx_hat - vwy_hat )./(1/dt - nu*lap) ; % -uwx_hat - vwy_hat
        
        % laplacian(psi) = -w
        psi = -w_hat./lap; psi(1) = 0;
        
        % Compute velocities
        u_hat =  1i*ky.*psi;
        v_hat = -1i*kx.*psi;
        
        % time update
        t = t + dt;
    end
    fprintf('t = %.3f\n',t);
    imagesc(x,y,real(ifft2(w_hat))')
    set(gca,'Ydir','normal'); grid minor;
    caxis([-1 1]*5)
    drawnow;
end

