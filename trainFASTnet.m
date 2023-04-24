function [trainingInfo,agent] = trainFASTnet(Neurons,batchsize,learningRateActor,learningRateCritic,Cpitch,Cmoment,SeqLength,Ts,MaxEpisodes,MaxSteps)
%% Train FASTnet 
%-----------------------------------------------------------------------------------------------------------------------------------------------%
% Current implementation is to minimize tower base moments with individual
% blade pitch control implemented through SIMULINK and the FAST S-Func
% created by NREL
%-----------------------------------------------------------------------------------------------------------------------------------------------%
%% Assign required variables:
FAST_InputFileName = '5MW_OC4Semi_WSt_WavesWN.fst';
TMax               = 500; % seconds
Cpitch = Cpitch ;
Cmoment = Cmoment ;
%% Specify environment options
nObsStates = 6;
obsInfo = rlNumericSpec([nObsStates 1]) ;
nActStates = 3 ; 
actInfo = rlNumericSpec([nActStates 1],"UpperLimit",1,"LowerLimit",-1);
env = rlSimulinkEnv("FAST_RL_Env", "FAST_RL_Env/controller",obsInfo,actInfo) ;
env.UseFastRestart = 'off' ;
obsPath = sequenceInputLayer(prod(obsInfo.Dimension),Name="netOin");
actPath = sequenceInputLayer(prod(actInfo.Dimension),Name="netAin");
%% Create main critic network
%  Previous versions included hidden layers before addition of the actor
%  network, however, it was not determined if this was necessary and
%  increased computation cost. 
%-----------------------------------------------------------------------------------------------------------------------------------------------%

commonPath = [
    concatenationLayer(1,2,Name="cat")
    lstmLayer(Neurons)
    lstmLayer(Neurons)
    lstmLayer(Neurons)
    reluLayer
    fullyConnectedLayer(1)
    ];

% Add paths to layerGraph network
criticNet = layerGraph(obsPath);
criticNet = addLayers(criticNet, actPath);
criticNet = addLayers(criticNet, commonPath);
% Connect paths
criticNet = connectLayers(criticNet,"netOin","cat/in1");
criticNet = connectLayers(criticNet,"netAin","cat/in2");
criticNet = dlnetwork(criticNet);
critic = rlQValueFunction(criticNet,obsInfo,actInfo,...
    ObservationInputNames="netOin",ActionInputNames="netAin");
%% Actor Net
actorNet = [
    sequenceInputLayer(prod(obsInfo.Dimension))
    lstmLayer(Neurons)
    lstmLayer(Neurons)
    lstmLayer(Neurons)
    reluLayer
    fullyConnectedLayer(prod(actInfo.Dimension)) 
    ];
actorNet = dlnetwork(actorNet);
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);
criticOpts = rlOptimizerOptions(LearnRate=learningRateCritic,GradientThreshold=1);
actorOpts = rlOptimizerOptions(LearnRate=learningRateActor,GradientThreshold=1);
agentOpts = rlDDPGAgentOptions(...
    SampleTime=Ts,...
    TargetSmoothFactor=1e-3,...
    ExperienceBufferLength=1e6,...
    DiscountFactor=0.99,...
    SequenceLength=SeqLength,...
    MiniBatchSize=batchsize, ...
    CriticOptimizerOptions=criticOpts, ...
    ActorOptimizerOptions=actorOpts);
agent = rlDDPGAgent(actor,critic,agentOpts) ;
%agent.UseExplorationPolicy = 1;
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',MaxEpisodes,...
    'MaxStepsPerEpisode',MaxSteps,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeCount',...
    'StopTrainingValue',1000,...
    'ScoreAveragingWindowLength',100,...
    'SaveAgentCriteria',"EpisodeCount",...
    'SaveAgentValue',250,Plots="training-progress");
trainingInfo = train(agent,env,trainOpts) ;
%% Important Functions
%  These will be useful in visualizing control surface
%       getAction(agent,{rand(obsInfo.Dimension)})
%       [action,state] = getAction(agent, ...
%       {rand([obsInfo.Dimension 1 9])});
end