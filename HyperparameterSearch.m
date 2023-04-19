clc; clear all; close all ;
%% Hyperparameter search of pitch control for a floating wind turbine
% Format for function is as follows:
% [trainingInfo,agent] = trainAgent2Hidden(Neurons,batchsize,learningRate,Cpitch,Cmoment,Ts,NumStepsAhead,DiscountFactor,MaxEpisodes,MaxSteps)
%-----------------------------------------------------------------------------------------------------------------------------------------------%
%% First training involves 128 neurons per layer with 2 hidden layers in each the actor and critic, learning rate of 0.005
%  pitch and moment coefficients of 0.5 and 2.5 respectively, a sample time
%  of 0.1 seconds, agent is looking 50 steps ahead to maximize reward at
%  each step with a discount factor of 0.99. Hyperparameter search involves
%  500 episodes with the agent having 30 seconds to minimize loads
[trainingInfo_1,agent_1] = trainAgent2Hidden(128,128,0.005,0.5,2.5,0.1,50,0.99,500,300) ;
