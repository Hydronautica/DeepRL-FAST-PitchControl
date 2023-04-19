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
