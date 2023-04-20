clc; clear all; close all ;
%% Hyperparameter search of pitch control for a floating wind turbine
% Format for function is as follows:
% [trainingInfo,agent] = trainAgent2Hidden(Neurons,batchsize,learningRate,Cpitch,Cmoment,Ts,NumStepsAhead,DiscountFactor,MaxEpisodes,MaxSteps)
%-----------------------------------------------------------------------------------------------------------------------------------------------%
FAST_InputFileName = '5MW_OC4Semi_WSt_WavesWN.fst';
TMax               = 500; % seconds
%% First training involves 64 neurons per layer with 2 hidden layers in each the actor and critic, learning rate of 0.005
%  pitch and moment coefficients of 0.5 and 2.5 respectively, a sample time
%  of 0.1 seconds, agent is looking 50 steps ahead to maximize reward at
%  each step with a discount factor of 0.99. Hyperparameter search involves
%  500 episodes with the agent having 30 seconds to minimize loads
%-----------------------------------------------------------------------------------------------------------------------------------------------%
Neurons            = 64 ;      % Number of neurons per layer
batchsize          = 128 ;     % Batch size
learningRate       = 0.005 ;   % Learning rate of both actor and critic
Cpitch             = 0.5 ;     % Pitch actuation cost coefficient
Cmoment            = 2.5 ;     % Moment cost coefficient
Ts                 = 0.1 ;     % Sampling time (s)
NumStepsAhead      = 50 ;      % Steps to look into future to predict reward
DiscountFactor     = 0.99 ;    % Specify importance of future estimates of reward (decay rate)
MaxEpisodes        = 500 ;     % Maximum number of episodes
MaxSteps           = 300 ;     % Maximum number of time steps per episode
MaxMoment          = 100000 ;  % Moment to normalize observations by
[trainingInfo_1, agent_1] = trainAgent3Hidden(Neurons,batchsize,learningRate,Cpitch,Cmoment,Ts,NumStepsAhead,DiscountFactor,MaxEpisodes,MaxSteps) ;
%% Second training involves 128 neurons per layer with 2 hidden layers in each the actor and critic, learning rate of 0.005
%  pitch and moment coefficients of 0.5 and 2.5 respectively, a sample time
%  of 0.1 seconds, agent is looking 50 steps ahead to maximize reward at
%  each step with a discount factor of 0.99. Hyperparameter search involves
%  500 episodes with the agent having 30 seconds to minimize loads
%-----------------------------------------------------------------------------------------------------------------------------------------------%
Neurons            = 128 ;     % Number of neurons per layer
batchsize          = 128 ;     % Batch size
learningRate       = 0.005 ;   % Learning rate of both actor and critic
Cpitch             = 0.5 ;     % Pitch actuation cost coefficient
Cmoment            = 2.5 ;     % Moment cost coefficient
Ts                 = 0.1 ;     % Sampling time (s)
NumStepsAhead      = 50 ;      % Steps to look into future to predict reward
DiscountFactor     = 0.99 ;    % Specify importance of future estimates of reward (decay rate)
MaxEpisodes        = 500 ;     % Maximum number of episodes
MaxSteps           = 300 ;     % Maximum number of time steps per episode
MaxMoment          = 100000 ;  % Moment to normalize observations by
[trainingInfo_2, agent_2] = trainAgent2Hidden(Neurons,batchsize,learningRate,Cpitch,Cmoment,Ts,NumStepsAhead,DiscountFactor,MaxEpisodes,MaxSteps) ;
%% Third training involves 256 neurons per layer with 2 hidden layers in each the actor and critic, learning rate of 0.005
%  pitch and moment coefficients of 0.5 and 2.5 respectively, a sample time
%  of 0.1 seconds, agent is looking 50 steps ahead to maximize reward at
%  each step with a discount factor of 0.99. Hyperparameter search involves
%  500 episodes with the agent having 30 seconds to minimize loads
%-----------------------------------------------------------------------------------------------------------------------------------------------%
Neurons            = 256 ;     % Number of neurons per layer
batchsize          = 128 ;     % Batch size
learningRate       = 0.005 ;   % Learning rate of both actor and critic
Cpitch             = 0.5 ;     % Pitch actuation cost coefficient
Cmoment            = 2.5 ;     % Moment cost coefficient
Ts                 = 0.1 ;     % Sampling time (s)
NumStepsAhead      = 50 ;      % Steps to look into future to predict reward
DiscountFactor     = 0.99 ;    % Specify importance of future estimates of reward (decay rate)
MaxEpisodes        = 500 ;     % Maximum number of episodes
MaxSteps           = 300 ;     % Maximum number of time steps per episode
MaxMoment          = 100000 ;  % Moment to normalize observations by
[trainingInfo_3, agent_3] = trainAgent2Hidden(Neurons,batchsize,learningRate,Cpitch,Cmoment,Ts,NumStepsAhead,DiscountFactor,MaxEpisodes,MaxSteps) ;
%% Fourth training involves 64 neurons per layer with 3 hidden layers in each the actor and critic, learning rate of 0.005
%  pitch and moment coefficients of 0.5 and 2.5 respectively, a sample time
%  of 0.1 seconds, agent is looking 50 steps ahead to maximize reward at
%  each step with a discount factor of 0.99. Hyperparameter search involves
%  500 episodes with the agent having 30 seconds to minimize loads
%-----------------------------------------------------------------------------------------------------------------------------------------------%
Neurons            = 64 ;     % Number of neurons per layer
batchsize          = 128 ;     % Batch size
learningRate       = 0.005 ;   % Learning rate of both actor and critic
Cpitch             = 0.5 ;     % Pitch actuation cost coefficient
Cmoment            = 2.5 ;     % Moment cost coefficient
Ts                 = 0.1 ;     % Sampling time (s)
NumStepsAhead      = 50 ;      % Steps to look into future to predict reward
DiscountFactor     = 0.99 ;    % Specify importance of future estimates of reward (decay rate)
MaxEpisodes        = 500 ;     % Maximum number of episodes
MaxSteps           = 300 ;     % Maximum number of time steps per episode
MaxMoment          = 100000 ;  % Moment to normalize observations by
[trainingInfo_4, agent_4] = trainAgent3Hidden(Neurons,batchsize,learningRate,Cpitch,Cmoment,Ts,NumStepsAhead,DiscountFactor,MaxEpisodes,MaxSteps) ;
%% Fifth training involves 128 neurons per layer with 3 hidden layers in each the actor and critic, learning rate of 0.005
%  pitch and moment coefficients of 0.5 and 2.5 respectively, a sample time
%  of 0.1 seconds, agent is looking 50 steps ahead to maximize reward at
%  each step with a discount factor of 0.99. Hyperparameter search involves
%  500 episodes with the agent having 30 seconds to minimize loads
%-----------------------------------------------------------------------------------------------------------------------------------------------%
Neurons            = 128 ;     % Number of neurons per layer
batchsize          = 128 ;     % Batch size
learningRate       = 0.005 ;   % Learning rate of both actor and critic
Cpitch             = 0.5 ;     % Pitch actuation cost coefficient
Cmoment            = 2.5 ;     % Moment cost coefficient
Ts                 = 0.1 ;     % Sampling time (s)
NumStepsAhead      = 50 ;      % Steps to look into future to predict reward
DiscountFactor     = 0.99 ;    % Specify importance of future estimates of reward (decay rate)
MaxEpisodes        = 500 ;     % Maximum number of episodes
MaxSteps           = 300 ;     % Maximum number of time steps per episode
MaxMoment          = 100000 ;  % Moment to normalize observations by
[trainingInfo_5, agent_5] = trainAgent3Hidden(Neurons,batchsize,learningRate,Cpitch,Cmoment,Ts,NumStepsAhead,DiscountFactor,MaxEpisodes,MaxSteps) ;
%% Sixth training involves 256 neurons per layer with 3 hidden layers in each the actor and critic, learning rate of 0.005
%  pitch and moment coefficients of 0.5 and 2.5 respectively, a sample time
%  of 0.1 seconds, agent is looking 50 steps ahead to maximize reward at
%  each step with a discount factor of 0.99. Hyperparameter search involves
%  500 episodes with the agent having 30 seconds to minimize loads
%-----------------------------------------------------------------------------------------------------------------------------------------------%
Neurons            = 256 ;     % Number of neurons per layer
batchsize          = 128 ;     % Batch size
learningRate       = 0.005 ;   % Learning rate of both actor and critic
Cpitch             = 0.5 ;     % Pitch actuation cost coefficient
Cmoment            = 2.5 ;     % Moment cost coefficient
Ts                 = 0.1 ;     % Sampling time (s)
NumStepsAhead      = 50 ;      % Steps to look into future to predict reward
DiscountFactor     = 0.99 ;    % Specify importance of future estimates of reward (decay rate)
MaxEpisodes        = 500 ;     % Maximum number of episodes
MaxSteps           = 300 ;     % Maximum number of time steps per episode
MaxMoment          = 100000 ;  % Moment to normalize observations by
[trainingInfo_6, agent_6] = trainAgent3Hidden(Neurons,batchsize,learningRate,Cpitch,Cmoment,Ts,NumStepsAhead,DiscountFactor,MaxEpisodes,MaxSteps) ;
%% Plot results
Ep        = trainingInfo_1.EpisodeIndex ;
% Average Rewards
T1_avgRew = trainingInfo_1.AverageReward ;
T2_avgRew = trainingInfo_2.AverageReward ;
T3_avgRew = trainingInfo_3.AverageReward ;
T4_avgRew = trainingInfo_4.AverageReward ;
T5_avgRew = trainingInfo_5.AverageReward ;
T6_avgRew = trainingInfo_6.AverageReward ;
% Estimated Reward
T1_estRew = trainingInfo_1.EpisodeQ0 ;
T2_estRew = trainingInfo_2.EpisodeQ0 ;
T3_estRew = trainingInfo_3.EpisodeQ0 ;
T4_estRew = trainingInfo_4.EpisodeQ0 ;
T5_estRew = trainingInfo_5.EpisodeQ0 ;
T6_estRew = trainingInfo_6.EpisodeQ0 ;
% Calculate Loss
Loss1 = T1_avgRew - T1_estRew ;
Loss2 = T2_avgRew - T2_estRew ;
Loss3 = T3_avgRew - T3_estRew ;
Loss4 = T4_avgRew - T4_estRew ;
Loss5 = T5_avgRew - T5_estRew ;
Loss6 = T6_avgRew - T6_estRew ;
% Create Figures
figure
subplot(2,1,1)
plot(Ep,T1_avgRew,Ep,T2_avgRew,Ep,T3_avgRew,Ep,T4_avgRew,Ep,T5_avgRew,Ep,T6_avgRew)
grid on
title('Average Reward')
ylabel('Average Reward')

axis([0 500 -1800 -700])
legend '64N-2HL' '128N-2HL' '256N-2HL' '64N-3HL' '128N-3HL' '256N-3HL'
subplot(2,1,2)
plot(Ep,-Loss1,Ep,-Loss2,Ep,-Loss3,Ep,-Loss4,Ep,-Loss5,Ep,-Loss6)
grid on
title('Critic Loss')
ylabel('Loss')
xlabel('Episode Number')
axis([0 500 500 1500])
sgtitle('Neuron and Layer Hyperparameter Search')






