function [trainingInfo,agent] = trainAgent2Hidden(Neurons,batchsize,learningRate,Cpitch,Cmoment,Ts,NumStepsAhead,DiscountFactor,MaxEpisodes,MaxSteps)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Cpitch = Cpitch ;
Cmoment = Cmoment ;
nObsStates = 4;
obsInfo = rlNumericSpec([nObsStates 1]) ;
MaxMoment = 100000 ;
nActStates = 3 ; % 3 Mooring Lines (3 change in lengths)
actInfo = rlNumericSpec([nActStates 1],"UpperLimit",1,"LowerLimit",-1);
env = rlSimulinkEnv("FAST_RL_Env", "FAST_RL_Env/controller",obsInfo,actInfo) ;
env.UseFastRestart = 'off' ;
actnet = [featureInputLayer(nObsStates,"Name","obs"),fullyConnectedLayer(Neurons,"Name",'fc1'),reluLayer("Name",'relu1'),fullyConnectedLayer(Neurons,"Name",'fc2'),reluLayer("Name",'relu2'),fullyConnectedLayer(nActStates,"Name","act"),tanhLayer("Name","tanh"),scalingLayer("Name","scact",Scale=max(actInfo.UpperLimit))] ;
actor = rlDeterministicActorRepresentation(actnet,obsInfo,actInfo,"Observation",'obs','Action','scact') ;
statePath = [
    featureInputLayer( ...
        obsInfo.Dimension(1), ...
        Name="obsPathInputLayer")
    fullyConnectedLayer(Neurons)
    reluLayer
    fullyConnectedLayer(Neurons)
    reluLayer
    fullyConnectedLayer(Neurons,Name="spOutLayer")
    ];

% Define action path
actionPath = [
    featureInputLayer( ...
        actInfo.Dimension(1), ...
        Name="actPathInputLayer")
    fullyConnectedLayer(Neurons, ...
        Name="HiddenActor1", ...
        BiasLearnRateFactor=0)
    fullyConnectedLayer(Neurons, ...
        Name="apOutLayer", ...
        BiasLearnRateFactor=0)
    ];

% Define common path
commonPath = [
    additionLayer(2,Name="add")
    reluLayer
    fullyConnectedLayer(Neurons)
    reluLayer
    fullyConnectedLayer(Neurons)
    reluLayer
    fullyConnectedLayer(1)
    ];

% Create layergraph, add layers and connect them
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,"spOutLayer","add/in1");
criticNetwork = connectLayers(criticNetwork,"apOutLayer","add/in2");
critic = rlQValueFunction(criticNetwork, ...
    obsInfo,actInfo, ...
    ObservationInputNames="obsPathInputLayer", ...
    ActionInputNames="actPathInputLayer");
criticOpts = rlOptimizerOptions(LearnRate=learningRate,GradientThreshold=1);
actorOpts = rlOptimizerOptions(LearnRate=learningRate,GradientThreshold=1);
agentOpts = rlDDPGAgentOptions(...
                'ActorOptimizerOptions',actorOpts,...
                'CriticOptimizerOptions',criticOpts,...
                'SampleTime',Ts,...
                'MiniBatchSize',batchsize,...
                'NumStepsToLookAhead',NumStepsAhead,...
                'DiscountFactor',DiscountFactor);
agent = rlDDPGAgent(actor,critic,agentOpts) ;
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',MaxEpisodes,...
    'MaxStepsPerEpisode',MaxSteps,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',200000,...
    'ScoreAveragingWindowLength',5,...
    'SaveAgentCriteria',"EpisodeReward",...
    'SaveAgentValue',5000,Plots="training-progress");

trainingInfo = train(agent,env,trainOpts) ;
end