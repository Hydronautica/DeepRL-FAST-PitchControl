clc; clear all; close all ;
%% Hyperparameter search of pitch control for a floating wind turbine
% Format for function is as follows:
% [trainingInfo,agent] = trainAgent2Hidden(Neurons,batchsize,learningRate,Cpitch,Cmoment,Ts,NumStepsAhead,DiscountFactor,MaxEpisodes,MaxSteps)
%-----------------------------------------------------------------------------------------------------------------------------------------------%
FAST_InputFileName = '5MW_OC4Semi_WSt_WavesWN.fst';
TMax               = 500; % seconds
load NoControlzp01.mat
%% First training involves 64 neurons per layer with 2 hidden layers in each the actor and critic, learning rate of 0.005
%  pitch and moment coefficients of 0.5 and 2.5 respectively, a sample time
%  of 0.1 seconds, agent is looking 50 steps ahead to maximize reward at
%  each step with a discount factor of 0.99. Hyperparameter search involves
%  500 episodes with the agent having 30 seconds to minimize loads
%-----------------------------------------------------------------------------------------------------------------------------------------------%
Neurons            = 400 ;      % Number of neurons per layer
batchsize          = 64 ;     % Batch size
NLactor            = 2  ;     % Number of hidden layers in actor
NLcritic           = 2  ;     % Number of hidden layers in critic
learningRateActor       = 0.0001 ;   % Learning rate of actor 
learningRateCritic      = 0.0001 ;   % Learning rate of criticCpitch  
Cpitch             = 0.01 ;     % Pitch actuation cost coefficient
Cmoment            = 2 ;     % Moment cost coefficient
Ts                 = 0.05 ;     % Sampling time (s)
SeqLength          = 20 ;         % Number of time samples to incorporate in LSTM cells
MaxEpisodes        = 2500 ;     % Maximum number of episodes
MaxSteps           = 600 ;     % Maximum number of time steps per episode
MaxMoment          = 100000 ;  % Moment to normalize observations by
ExVar              = 0.01 ;         % Actor exploration variance (percent/100)
[trainingInfo_1, agent_1] = trainFASTnet(Neurons,NLactor,NLcritic,batchsize,learningRateActor,learningRateCritic,Cpitch,Cmoment,SeqLength,Ts,ExVar,MaxEpisodes,MaxSteps) ;
%% Plot results
Ep        = trainingInfo_1.EpisodeIndex ;
% Average Rewards
T1_avgRew = trainingInfo_1.EpisodeReward ;

% Estimated Reward
T1_estRew = trainingInfo_1.EpisodeQ0 ;

% Calculate Loss
Loss1 = T1_avgRew - T1_estRew ;

% Create Figures
figure
subplot(2,1,1)
plot(Ep,T1_avgRew)
grid on
title('Average Reward')
ylabel('Average Reward')

%axis([0 500 -0 1e6])
legend '128N3H128-LR1' 
subplot(2,1,2)
plot(Ep,-Loss1)
grid on
title('Critic Loss')
ylabel('Loss')
xlabel('Episode Number')
%axis([0 500 500 1500])
sgtitle('Learning Rate Hyperparameter Search')

%agent.UseExplorationPolicy = true ;




