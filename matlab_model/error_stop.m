mdl = "hydraulic_model";
load_system(mdl);

set_param(mdl, 'SimulationCommand','stop');
set_param(mdl, 'FastRestart', 'off');
clear;
clc;
% close_system(mdl, 0);
