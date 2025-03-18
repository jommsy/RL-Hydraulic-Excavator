%% Time Setting
maxStep = 5e-5;
tEnd = 25;
%% Initial Conditions
initBoomHEPr = 10; % [bar]
initBoomREPr = 10; % [bar]
initStickHEPr = 10; % [bar]
initStickREPr = 10; % [bar]
initBucketHEPr = 10; % [bar] 
initBucketREPr = 10; % [bar]
p0 = 2;
% Gravity
gravity = -9.81; % [m/s^2]

%% Maximum Flow Rates

%% Pump
pumpSpeed = 2000; % [rpm]
pumpChamberVol = 1e-4; % [mm^3]

%% Pipes
pipeLength = 800; % [mm]
pipeHydraulicDiam = 40; % [mm]
pipeCrossSectArea = pi * (pipeHydraulicDiam/2)^2; % [mm^2] 
pipeChamberVol = 1e-3;  % [m^3]

%% Pump Pressure Relief Valves
pumpPRV.reliefPr = 245; % [bar]
pumpPRV.regRange = 5; % [bar]
pumpPRV.maxOpeningArea = 1200; % [mm^2]
pumpPRV.leakageArea = 0.001; % [mm^2]


%% Valve block
% Function Pressure Relief Valves
fcnPRV.reliefPr = 270; % [bar]
fcnPRV.regRange = 5; % [bar]
fcnPRV.maxOpeningArea = 1200; % [mm^2]
fcnPRV.leakageArea = 0.001; % [mm^2]

% Check Valves
checkValves.crackingPrDiff = 0.1; % [bar]
checkValves.maxOpeningPrDiff = 0.5; % [bar]
checkValves.maxOpeningArea = 1000; % [mm^2]
checkValves.leakageArea = 0.0001; % [mm^2]

% Valve Spool Vector
spoolPos = (0:0.1:5)'; % [mm] 

% Valve bypass area
BiPAreaSwing = [1e-4; 1e-4; 3.6.*(spoolPos(3:end))]; % [mm^2] 
BiPAreaBoom = [1e-4; 1e-4; 2.*(spoolPos(3:end))]; % [mm^2] 
BiPAreaArm = [1e-4; 1e-4; 1.6.*(spoolPos(3:end))];
BiPAreaBucket = [1e-4; 1e-4; 1.5.*(spoolPos(3:end))];
toTSideArea = [1e-4; 1e-4; 20.*spoolPos(3:end)]; % [mm^2]
pToSideArea = [1e-4; 1e-4; 100.*spoolPos(3:11); repmat(100, length(spoolPos) - 11, 1)];

% Swing
swingPRV.reliefPr = 160; % [bar]
swingPRV.regRange = 5; % [bar]
swingPRV.maxOpeningArea = 100; % [mm^2]
swingPRV.leakageArea = 0.001; % [mm^2]
swingValve.spoolPosVector = spoolPos; % [mm]
swingValve.BiPOrificeAreaVector = BiPAreaSwing; % [mm^2]
swingValve.PtoAOrificeAreaVector = pToSideArea; % [mm^2]
swingValve.AtoTOrificeAreaVector = toTSideArea; % [mm^2]
swingValve.BtoTOrificeAreaVector = toTSideArea; % [mm^2]
swingValve.PtoBOrificeAreaVector = pToSideArea; % [mm^2]

% Boom
boomValve.spoolPosVector = spoolPos; % [mm]
boomValve.BiPOrificeAreaVector = BiPAreaBoom; % [mm^2]
boomValve.PtoAOrificeAreaVector = pToSideArea; % [mm^2]
boomValve.AtoTOrificeAreaVector = toTSideArea; % [mm^2]
boomValve.BtoTOrificeAreaVector = toTSideArea; % [mm^2]
boomValve.PtoBOrificeAreaVector = pToSideArea; % [mm^2]

% Arm
stickValve.spoolPosVector = spoolPos; % [mm]
stickValve.BiPOrificeAreaVector = BiPAreaArm; % [mm^2]
stickValve.PtoAOrificeAreaVector = pToSideArea; % [mm^2]
stickValve.AtoTOrificeAreaVector = toTSideArea; % [mm^2]
stickValve.BtoTOrificeAreaVector = toTSideArea; % [mm^2]
stickValve.PtoBOrificeAreaVector = pToSideArea; % [mm^2]

% Bucket
bucketValve.spoolPosVector = spoolPos; % [mm]
bucketValve.BiPOrificeAreaVector = BiPAreaBucket; % [mm^2]
bucketValve.PtoAOrificeAreaVector = pToSideArea; % [mm^2] 
bucketValve.AtoTOrificeAreaVector = toTSideArea; % [mm^2] 
bucketValve.BtoTOrificeAreaVector = toTSideArea; % [mm^2] 
bucketValve.PtoBOrificeAreaVector = pToSideArea; % [mm^2] 

%% Mechanics
% swing
swingDisplacement = 23.6; % [cc/rev]
swingGearRatio = 134; % Motor gearbox and pinion to ring combined 22.7 * 5.9 = 134

% boom
boomStroke = 650; % [mm] 
boomBoreDiameter = 63; % [mm]
boomRodDiameter = 30; % [mm]
boomHEArea = pi * (boomBoreDiameter/2)^2; % [mm^2]
boomREArea = pi * (boomBoreDiameter/2)^2 - pi*(boomRodDiameter/2)^2; % [mm^2]
boomCylFriction.brkawyToCoulRatio = 1.2; % 1.2 -> 2.4
boomCylFriction.brkawyVel = 1e-2; % [m/s]
boomCylFriction.preloadForce = 800; % [N] 20 -> 800
boomCylFriction.coulForceCoef = 1e-4; % [N/Pa]
boomCylFriction.viscCoeff = 5e3; % [N/(m/s)] 

% arm
stickStroke  = 780; % [mm] 
stickBoreDiameter = 55; % [mm]
stickRodDiameter = 30; % [mm]
stickHEArea = pi * (stickBoreDiameter/2)^2; % [mm^2]
stickREArea = pi * (stickBoreDiameter/2)^2 - pi*(stickRodDiameter/2)^2; % [mm^2]
stickCylFriction.brkawyToCoulRatio = 1.2;
stickCylFriction.brkawyVel = 1e-2; % [m/s]
stickCylFriction.preloadForce = 800; % [N] 20 -> 800
stickCylFriction.coulForceCoef = 1e-4; % [N/Pa]
stickCylFriction.viscCoeff = 5e3; % [N/(m/s)] 

% bucket
bucketStroke  = 530; % [mm] 
bucketBoreDiameter = 55; % [mm]
bucketRodDiameter = 30; % [mm]
bucketHEArea = pi * (bucketBoreDiameter/2)^2; % [mm^2]
bucketREArea = pi * (bucketBoreDiameter/2)^2 - pi*(bucketRodDiameter/2)^2; % [mm^2]
bucketCylFriction.brkawyToCoulRatio = 1.2;
bucketCylFriction.brkawyVel = 1e-2; % [m/s]
bucketCylFriction.preloadForce = 800; % [N] 20 -> 800
bucketCylFriction.coulForceCoef = 1e-4; % [N/Pa]
bucketCylFriction.viscCoeff = 5e3; % [N/(m/s)]

% Joint bound
% lowerSwing = - 0.75 * pi;
% upperSwing =  0.75 * pi;
% lowerBoom = -0.15;
% upperBoom = 0.17;
% lowerArm = -0.125;
% upperArm = 0.33;
% lowerBucket = -0.15;
% upperBucket = 0.20;

%% Initial position state
% initSwingAngle = 0; % [deg]
% initBoomPistonPos = 450; % [mm]
% initStickPistonPos = 450; % [mm]
% initBucketPistonPos = 250; % [mm]

%% start model and set as faster restart
mdl = "hydraulic_model";
load_system(mdl);
set_param(mdl, 'SimulationMode', 'accelerator');
% set_param(mdl, 'FastRestart', 'on');
set_param(mdl+'/Excavator/Timer/Tmp', 'Value', num2str(0));
set_param(mdl+'/Excavator/spoolPos', 'Value', mat2str([0 0 0 0]));
set_param(mdl+'/Excavator/loads', 'Value', mat2str([0 0 0 0]));
set_param(mdl,'SimulationCommand','start',...
    'SimulationCommand','pause');










