mdl = "hydraulic_model";
set_param(mdl, 'SimulationCommand', 'stop');
% clear sim_step;
set_param(mdl+'/Excavator/Timer/Tmp', 'Value', num2str(0));
set_param(mdl+'/Excavator/Timer/Ts', 'value', num2str(0));
set_param(mdl+'/Excavator/spoolPos', 'Value', mat2str([0 0 0 0]));
set_param(mdl+'/Excavator/loads', 'Value', mat2str([0 0 0 0]));
set_param(mdl,'SimulationCommand','start',...
    'SimulationCommand','pause');
