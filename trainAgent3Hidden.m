function [trainingInfo,agent] = trainAgent3Hidden(Neurons,batchsize,learningRateActor,learningRateCritic,Cpitch,Cmoment,Ts,NumStepsAhead,DiscountFactor,MaxEpisodes,MaxSteps)
FAST_InputFileName = '5MW_OC4Semi_WSt_WavesWN.fst';
TMax               = 500; % seconds
Cpitch = Cpitch ;
Cmoment = Cmoment ;
nObsStates = 7;
obsInfo = rlNumericSpec([nObsStates 1]) ;
MaxMoment = 100000 ;
nActStates = 3 ; % 3 Mooring Lines (3 change in lengths)
actInfo = rlNumericSpec([nActStates 1],"UpperLimit",1,"LowerLimit",-1);
env = rlSimulinkEnv("FAST_RL_Env", "FAST_RL_Env/controller",obsInfo,actInfo) ;
env.UseFastRestart = 'off' ;
actnet = [sequenceInputLayer(nObsStates,"Name","obs"),lstmLayer(Neurons,"Name",'fc1'),fullyConnectedLayer(Neurons,"Name",'fc2'),reluLayer("Name",'relu2'),fullyConnectedLayer(nActStates,"Name","act"),tanhLayer("Name","tanh"),scalingLayer("Name","scact",Scale=max(actInfo.UpperLimit))] ;
actor = rlDeterministicActorRepresentation(actnet,obsInfo,actInfo,"Observation",'obs','Action','scact') ;
statePath = [
    sequenceInputLayer( ...
        obsInfo.Dimension(1), ...
        Name="obsPathInputLayer")
    lstmLayer(Neurons)
    fullyConnectedLayer(Neurons)
    reluLayer
    fullyConnectedLayer(Neurons)
    reluLayer
    fullyConnectedLayer(Neurons,Name="spOutLayer")
    ];

% Define action path
actionPath = [
    sequenceInputLayer( ...
        actInfo.Dimension(1), ...
        Name="actPathInputLayer")
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
criticOpts = rlOptimizerOptions(LearnRate=learningRateCritic,GradientThreshold=1,L2RegularizationFactor=1e-4);
actorOpts = rlOptimizerOptions(LearnRate=learningRateActor,GradientThreshold=1,L2RegularizationFactor=1e-4);
agentOpts = rlDDPGAgentOptions(...
                'ActorOptimizerOptions',actorOpts,...
                'CriticOptimizerOptions',criticOpts,...
                'SampleTime',Ts,...
                'MiniBatchSize',batchsize,...
                'NumStepsToLookAhead',NumStepsAhead,...
                'ExperienceBufferLength',1e6,...
                'SequenceLength',10,...
                'DiscountFactor',DiscountFactor);
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
    'SaveAgentValue',1,Plots="training-progress");

trainingInfo = train(agent,env,trainOpts) ;
end