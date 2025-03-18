function [pos, vel] = sim_step(com_timestep, eng_time, spoolPos, loads)
    % FOR NORMAL USING
    % persistent isFirstRun;

    mdl = "hydraulic_model";
    set_param(mdl+'/Excavator/Timer/Ts', 'value', num2str(com_timestep));
    set_param(mdl+'/Excavator/spoolPos', 'value', mat2str(spoolPos));
    set_param(mdl+'/Excavator/loads', 'value', mat2str(loads));

    % if isempty(isFirstRun)
    %     set_param(mdl, 'SimulationCommand', 'start');
    %     isFirstRun = false;
    % else
    set_param(mdl, 'SimulationCommand', 'continue');
    % end

    % 避免时序错误
    max_attempts = 100; % 设置最大尝试次数，防止无限循环
    attempt = 0;

    while attempt < max_attempts
        attempt = attempt + 1;
        pause(0.002);
        if evalin('base', 'exist(''out'', ''var'')')
            out = evalin('base', 'out');
            if out.tout(end) > eng_time
                % index = find (floor(1000 .*(out.tout)) == floor(1000 * eng_time));
                index = find (out.tout >= eng_time);
                pos = [out.angSwing(index(1)), ... 
                        out.posBoom(index(1)),...
                        out.posArm(index(1)), ...
                        out.posBucket(index(1))];
                vel = [out.velSwing(index(1)), ... 
                        out.velBoom(index(1)),...
                        out.velArm(index(1)), ...
                        out.velBucket(index(1))];
                pos = round(pos, 5);
                vel = round(vel, 5);
                break
            end
        end
    end
    
    if attempt >= max_attempts
        pos = [];
        vel = [];
    end

    % error('Failed to get the expected output from Simulink model within the maximum attempts.');
    % sm = evalin('base', 'sm');
    % sm = setVariable(sm,'motor_speed',motor_speed,'Workspace','hydraulic_model');
    % sm = setVariable(sm,'load_boom',load_boom,'Workspace','hydraulic_model');
    % step(sm);
    % force_boom = sm.SimulationOutput.force_boom(eng_step);
end
