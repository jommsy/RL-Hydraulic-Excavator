function init_condition(dispBias, initSimPos)
    % evalin('base', 'clear');
    assignin('base', 'initBoomPistonPos', initSimPos(1));
    assignin('base', 'initStickPistonPos', initSimPos(2));
    assignin('base', 'initBucketPistonPos', initSimPos(3));

    assignin('base', 'swingBias', dispBias(1));
    assignin('base', 'boomBias', dispBias(2));
    assignin('base', 'armBias', dispBias(3));
    assignin('base', 'bucketBias', dispBias(4));
    
    lowerSwing = - 0.75 * pi - dispBias(1);
    upperSwing =  0.75 * pi - dispBias(1);
    lowerBoom = -0.15 - dispBias(2);
    upperBoom = 0.17 - dispBias(2);
    lowerArm = -0.125 - dispBias(3);
    upperArm = 0.33 - dispBias(3);
    lowerBucket = -0.15 - dispBias(4);
    upperBucket = 0.20 - dispBias(4);
    
    assignin('base', 'lowerSwing', lowerSwing);
    assignin('base', 'upperSwing', upperSwing);
    assignin('base', 'lowerBoom', lowerBoom);
    assignin('base', 'upperBoom', upperBoom);
    assignin('base', 'lowerArm', lowerArm);
    assignin('base', 'upperArm', upperArm);
    assignin('base', 'lowerBucket', lowerBucket);
    assignin('base', 'upperBucket', upperBucket);
end
