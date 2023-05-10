function [trainingInfo,agent] = trainFASTnet(Neurons,NLactor,NLcritic,batchsize,learningRateActor,learningRateCritic,Cpitch,Cmoment,SeqLength,Ts,ExVar,MaxEpisodes,MaxSteps)
%% Train FASTnet 
%-----------------------------------------------------------------------------------------------------------------------------------------------%
% Current implementation is to minimize tower base moments with individual
% blade pitch control implemented through SIMULINK and the FAST S-Func
% created by NREL
%-----------------------------------------------------------------------------------------------------------------------------------------------%
%% Assign required variables:
rng(2) ;
FAST_InputFileName = '5MW_OC4Semi_WSt_WavesWN.fst';
TMax               = 3600; % seconds
Cpitch = Cpitch ;
Cmoment = Cmoment ;
%% Specify environment options
nObsStates = 3;
obsInfo = rlNumericSpec([nObsStates 1]) ;
nActStates = 3 ; 
actInfo = rlNumericSpec([nActStates 1],"UpperLimit",2,"LowerLimit",-2);
env = rlSimulinkEnv("FAST_RL_Env", "FAST_RL_Env/controller",obsInfo,actInfo) ;
env.UseFastRestart = 'off' ;

% obsPath = sequenceInputLayer(prod(obsInfo.Dimension),Name="netOin");
% actPath = sequenceInputLayer(prod(actInfo.Dimension),Name="netAin");
%% Create main critic network
%  Previous versions included hidden layers before addition of the actor
%  network, however, it was not determined if this was necessary and
%  increased computation cost. 
%-----------------------------------------------------------------------------------------------------------------------------------------------%
% commonPath = [
%         concatenationLayer(1,2,Name="cat")
%         lstmLayer(64)
%         lstmLayer(64)
%         reluLayer
%         fullyConnectedLayer(Neurons)
%         reluLayer
%         fullyConnectedLayer(Neurons-100)
%         reluLayer
%         fullyConnectedLayer(1)
%         ];
% mainPath = [
%         featureInputLayer(prod(obsInfo.Dimension),Name="obsInLyr")
%         additionLayer(2,Name="add")
%         reluLayer
%         fullyConnectedLayer(Neurons)
%         reluLayer
%         fullyConnectedLayer(Neurons-100)
%         reluLayer
%         fullyConnectedLayer(1)
%         ];
% 
% actionPath = [
%     featureInputLayer(prod(actInfo.Dimension),Name="actOutLyr")
%     % fullyConnectedLayer(Neurons,Name="actOutLyr")
%     ];
% criticNet = layerGraph(mainPath);
% criticNet = addLayers(criticNet,actionPath);    
% criticNet = connectLayers(criticNet,"actOutLyr","add/in2");
% % Add paths to layerGraph network
% criticNet = layerGraph(obsPath);
% criticNet = addLayers(criticNet, actPath);
% criticNet = addLayers(criticNet, commonPath);
% % Connect paths
% criticNet = connectLayers(criticNet,"netOin","add/in1");
% criticNet = connectLayers(criticNet,"netAin","add/in2");
% % criticNet = connectLayers(criticNet,"netOin","cat/in1");
% % criticNet = connectLayers(criticNet,"netAin","cat/in2");
% mainPath = [
%     featureInputLayer(prod(obsInfo.Dimension),Name="obsInLyr")
%     additionLayer(2,Name="add")
%     reluLayer
%     fullyConnectedLayer(Neurons)
%     reluLayer
%     fullyConnectedLayer(Neurons-100)
%     reluLayer
%     fullyConnectedLayer(1,Name="QValLyr")
%     ];
% 
% % Action path
% actionPath = [
%     featureInputLayer(prod(actInfo.Dimension),Name="actInLyr")
%     fullyConnectedLayer(prod(obsInfo.Dimension),Name="actOutLyr",BiasLearnRateFactor=0)
%     ];
% 
% % Assemble layergraph object
% criticNet = layerGraph(mainPath);
% criticNet = addLayers(criticNet,actionPath);    
% criticNet = connectLayers(criticNet,"actOutLyr","add/in2");
% criticNet = dlnetwork(criticNet);
% critic = rlQValueFunction(criticNet,obsInfo,actInfo,...
%     ObservationInputNames="obsInLyr",ActionInputNames="actInLyr");
statePath = [
    featureInputLayer( ...
        obsInfo.Dimension(1), ...
        Name="obsPathInputLayer")
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300,Name="spOutLayer")
    ];

% Define action path
actionPath = [
    featureInputLayer( ...
        actInfo.Dimension(1), ...
        Name="actPathInputLayer")
    fullyConnectedLayer(300, ...
        Name="apOutLayer", ...
        BiasLearnRateFactor=0)
    ];

% Define common path
commonPath = [
    additionLayer(2,Name="add")
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
criticNetwork = dlnetwork(criticNetwork);
summary(criticNetwork)
critic = rlQValueFunction(criticNetwork, ...
    obsInfo,actInfo, ...
    ObservationInputNames="obsPathInputLayer", ...
    ActionInputNames="actPathInputLayer");
critic.UseDevice = "gpu" ;
%% Actor Net
    % actorNet = [
    %     sequenceInputLayer(prod(obsInfo.Dimension))
    %     lstmLayer(64)
    %     lstmLayer(64)
    %     reluLayer
    %     fullyConnectedLayer(Neurons)
    %     reluLayer
    %     fullyConnectedLayer(Neurons-100)
    %     reluLayer
    %     fullyConnectedLayer(prod(actInfo.Dimension)) 
    %     tanhLayer
    %     scalingLayer(Scale=max(actInfo.UpperLimit))
    %      ];
    % actorNet = [
    %     featureInputLayer(prod(obsInfo.Dimension))
    %     fullyConnectedLayer(Neurons)
    %     reluLayer
    %     fullyConnectedLayer(Neurons-100)
    %     reluLayer
    %     fullyConnectedLayer(prod(actInfo.Dimension)) 
    %     tanhLayer
    %     scalingLayer(Scale=max(actInfo.UpperLimit))
    %      ];
actorNetwork = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(3)
    tanhLayer
    scalingLayer(Scale=max(actInfo.UpperLimit))
    ];

actorNetwork = dlnetwork(actorNetwork);
summary(actorNetwork)
actor = rlContinuousDeterministicActor(actorNetwork,obsInfo,actInfo);
actor.UseDevice = "gpu" ;
criticOpts = rlOptimizerOptions(LearnRate=learningRateCritic,GradientThreshold=1);
actorOpts = rlOptimizerOptions(LearnRate=learningRateActor,GradientThreshold=1);
agentOpts = rlDDPGAgentOptions(...
    SampleTime=Ts,...
    TargetSmoothFactor=1e-3,...
    ExperienceBufferLength=1e4,...
    DiscountFactor=0.99,...
    NumStepsToLookAhead=50,...
    MiniBatchSize=batchsize, ...
    CriticOptimizerOptions=criticOpts, ...
    ActorOptimizerOptions=actorOpts);
Var = (actInfo.UpperLimit - actInfo.LowerLimit)*ExVar/sqrt(Ts) ;

agentOpts.NoiseOptions.Variance = Var;
%agentOpts.NoiseOptions.MeanAttractionConstant = 0.8;
%agentOpts.NoiseOptions.VarianceDecayRate = 0;
%agentOpts.NoiseOptions.StandardDeviationMin = 0.01;
agent = rlDDPGAgent(actor,critic,agentOpts) ;

%agent.UseExplorationPolicy = 1;
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',MaxEpisodes,...
    'MaxStepsPerEpisode',MaxSteps,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeCount',...
    'StopTrainingValue',5000,...
    'ScoreAveragingWindowLength',100,...
    'SaveAgentCriteria',"EpisodeReward",...
    'SaveAgentValue',-20,Plots="training-progress");
trainingInfo = train(agent,env,trainOpts) ;
%% Important Functions
%  These will be useful in visualizing control surface
%       getAction(agent,{rand(obsInfo.Dimension)})
%       [action,state] = getAction(agent, ...
%       {rand([obsInfo.Dimension 1 9])});
end