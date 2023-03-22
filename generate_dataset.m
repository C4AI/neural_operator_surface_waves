% Set tank size and number of grid nodes
x0 = -14.04/2;
x1 = 14.04/2;
nx = 201;

y0 = -14.04/2;
y1 = 14.04/2;
ny = 201;

% Set time interval and accomodation times

tf = 1;
ta_rand = 5;
ta_gauss = 1;

% Number of Fourrier modes for random initial conditions
nModes = 5;

% Settings for Gaussian initial conditions
nPeaks = 1;
minAlpha = 0.1;
maxAlpha = 1;

% Fraction of the dataset composed by Gaussian initial conditions
fracGauss = 0.3;

% Set tank parameters
pars.Lx = x1-x0;
pars.Ly = y1-y0;
pars.h = 4.1;
pars.g = 9.81;

% Set parameters for cossine transform
pars.m = nx;
pars.n = ny;

% Number of samples, one for each random seed
seeds = 1:100;
ns = length(seeds);

% Allocate variables
Y0 = nan(nx,ny,2,ns);
Y1 = nan(nx,ny,2,ns);

x = linspace(x0,x1,nx);
y = linspace(y0,y1,ny);

t_rand = [0 ta_rand ta_rand+tf];
t_gauss = [0 ta_gauss ta_gauss+tf];

% Iterate each sample
for i = 1:ns

    disp(i)

    if rand(1) > fracGauss
        Y = initialconditionRand([nx,ny],0.1,nModes,seeds(i));
        E = greenFunc(Y,t_rand,0,pars,false,true);
    else
        Y = initialconditionGauss(x,y,0.1,nPeaks,minAlpha,maxAlpha,seeds(i));
        E = greenFunc(Y,t_gauss,0,pars,false,true);
    end

    Y0(:,:,1,i) = E(:,:,2,1);
    Y0(:,:,2,i) = E(:,:,2,2);
    Y1(:,:,1,i) = E(:,:,3,1);
    Y1(:,:,2,i) = E(:,:,3,2);

end

% Save dataset
fileName = 'data/dataset.mat';

save(fileName,'Y0','Y1','x','y')

% Generators for the initial conditions
function Y = initialconditionRand(size,amplitude,nModes,seed)
    if nargin < 2
        amplitude = 1;
    end
    if nargin < 3
        nModes = 5;
    end
    if nargin == 4
        rng(seed);
    end

    A = randn(size);
    A(nModes+1:end-nModes,:) = 0;
    A(:,nModes+1:end-nModes) = 0;

    Y(:,:,1,1) = real(ifft2(A));
    Y(:,:,1,2) = imag(ifft2(A));

    % Neumann condition for height at the sides
    Y(:,1,:,:) = 4/3*Y(:,2,:,:) - 1/3*Y(:,3,:,:);
    Y(:,end,:,:) = 4/3*Y(:,end-1,:,:) - 1/3*Y(:,end-2,:,:);
    Y(1,:,:,:) = 4/3*Y(2,:,:,:) - 1/3*Y(3,:,:,:);
    Y(end,:,:,:) = 4/3*Y(end-1,:,:,:) - 1/3*Y(end-2,:,:,:);

    Y = Y/rms(Y(:))*amplitude;

end

function Y = initialconditionGauss(x,y,amplitude,nPeaks,minAlpha,maxAlpha,seed)
    if nargin == 6
        rng(seed);
    end

    Y = zeros(length(y),length(x),1,2);

    for i = 1:nPeaks
        x0 = x(1) + (x(end)-x(1))*rand(1);
        y0 = y(1) + (y(end)-y(1))*rand(1);
        a = minAlpha + (maxAlpha-minAlpha)*rand(1);

        Y = Y + exp(-a*((y'-y0).^2 + (x-x0).^2));
    end

    Y(:,:,:,2) = 0;

    % Neumann condition for height at the sides
    Y(:,1,:,:) = 4/3*Y(:,2,:,:) - 1/3*Y(:,3,:,:);
    Y(:,end,:,:) = 4/3*Y(:,end-1,:,:) - 1/3*Y(:,end-2,:,:);
    Y(1,:,:,:) = 4/3*Y(2,:,:,:) - 1/3*Y(3,:,:,:);
    Y(end,:,:,:) = 4/3*Y(end-1,:,:,:) - 1/3*Y(end-2,:,:,:);

    Y = Y/rms(Y(:))*amplitude;

end

% Green function
function e = greenFunc(e0,t,t0,pars,heavySide,returnElevation,inputElevation)

    % If true, flow is considered static before the initial time
    if ~exist('heavySide','var')
        heavySide = true;
    end

    % If true, the output will be transformed back from cossines to an array of elevations
    if ~exist('returnElevation','var')
        returnElevation = false;
    end

    % If true, the input is assumed to be in cossine form
    if ~exist('inputElevation','var')
        inputElevation = true;
    end
    
    % Checks whether eta_t was provided
    hasDeriv = size(e0,4)==2;
    
    nx = size(e0,2);
    ny = size(e0,1);
    
    % If needed, perform cossine transform
    t = permute(t,[1 3 2])-t0;
    if inputElevation
        at0 = cossineTransform(e0,pars.m,pars.n);
    else
        at0 = e0;
        nx = pars.m;
        ny = pars.n;
        pars.m = size(e0,2);
        pars.n = size(e0,1);
    end
    
    % Compute wavenumbers
    if isfield(pars,'Lx')
        k = sqrt(((0:pars.m-1)/pars.Ly).^2 + ((0:pars.n-1)'/pars.Lx).^2)*pi;
    else
        k = sqrt((0:pars.m-1).^2 + ((0:pars.n-1)').^2)*pi/pars.L;
    end
    
    % Compute wave velocities
    alpha = sqrt(pars.g*k.*tanh(pars.h*k));
    
    % Compute phase
    if hasDeriv
        phi = atan(- at0(:,:,:,2)./at0(:,:,:,1) ./ alpha);
        phi(isnan(phi)) = 0;
    else
        phi = 0;
    end
    
    a0 = at0(:,:,:,1)./cos(phi);
    
    % Analytical solution
    if hasDeriv
        a = cat(4,a0.*cos(alpha.*t+phi),-a0.*alpha.*sin(alpha.*t+phi));
    else
        a = a0.*cos(alpha.*t+phi);
    end
    
    if heavySide
        a(:,:,t<0) = 0;
    end

    % Inverse cossine transform
    if returnElevation
        e = cossineTransform(a,nx,ny,1);
    else
        e = a;
    end

end

% Cossine transform
function a = cossineTransform(e,m,n,invert)

if ~exist('invert','var') || ~invert
    nx = size(e,2);
    ny = size(e,1);
    if ~exist('m','var')
        m = nx;
    end
    if ~exist('n','var')
        n = ny;
    end
    cx = getCossines1(nx,m);
    cy = getCossines2(ny,n);

    a = zeros(n,m,size(e,3),size(e,4));
    for i = 1:size(e,3)
        for j = 1:size(e,4)
            temp = cx\e(:,:,i,j)';
            a(:,:,i,j) = cy\temp';
        end
    end
else
    nx = size(e,2);
    ny = size(e,1);
    if ~exist('m','var')
        m = nx;
    end
    if ~exist('n','var')
        n = ny;
    end
    cx = getCossines1(m,nx);
    cy = getCossines2(n,ny);

    a = zeros(n,m,size(e,3),size(e,4));
    for i = 1:size(e,3)
        for j = 1:size(e,4)
            temp = cx*e(:,:,i,j)';
            a(:,:,i,j) = cy*temp';
        end
    end
end

end

% Compute the cossines at the grid nodes, persistent variables are used to avoid repeating calculations
function c = getCossines1(nx,m)
    persistent nx_ m_ c_

    if ~isempty(nx_) && nx == nx_ && m == m_
        c = c_;
    else
        eta = linspace(0,1,nx)';
        i = (0:m-1);
        c = cos(pi*i.*eta);
        nx_ = nx;
        m_ = m;
        c_ = c;
    end
end

function c = getCossines2(nx,m)
    persistent nx_ m_ c_

    if ~isempty(nx_) && nx == nx_ && m == m_
        c = c_;
    else
        eta = linspace(0,1,nx)';
        i = (0:m-1);
        c = cos(pi*i.*eta);
        nx_ = nx;
        m_ = m;
        c_ = c;
    end
end