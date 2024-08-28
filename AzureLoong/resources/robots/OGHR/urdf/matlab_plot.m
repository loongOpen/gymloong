clear variables;
close all;
robot=importrobot('OGHR_wholeBody.urdf');
config=homeConfiguration(robot);
config(16).JointPosition=0;
show(robot,config,'visual','on','collision','off')
m=0;
for i=1:1:robot.NumBodies
    m=m+robot.Bodies{i}.Mass;
    fprintf("%s mass= %.3f\n",robot.Bodies{i}.Name,robot.Bodies{i}.Mass);
end

% <origin rpy="0.00000000 0.00000000 0.00000000" xyz="0.02500000 -0.03100000 -0.02600000"/>
%             <geometry>
%                 <cylinder radius="0.00800000" length="0.19800000"/>
%             </geometry>