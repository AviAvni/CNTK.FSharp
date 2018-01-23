(*
This file is intended to load dependencies in an F# script,
to train a model from the scripting environment.
CNTK, CPU only, is assumed to have been installed via Paket.
*)

open System
open System.IO
open System.Collections.Generic

Environment.SetEnvironmentVariable("Path",
    Environment.GetEnvironmentVariable("Path") + ";" + __SOURCE_DIRECTORY__)

let dependencies = [
        "./packages/CNTK.CPUOnly/lib/net45/x64/"
        "./packages/CNTK.CPUOnly/support/x64/Dependency/"
        "./packages/CNTK.CPUOnly/support/x64/Dependency/Release/"
        "./packages/CNTK.CPUOnly/support/x64/Release/"    
    ]

dependencies 
|> Seq.iter (fun dep -> 
    let path = Path.Combine(__SOURCE_DIRECTORY__,dep)
    Environment.SetEnvironmentVariable("Path",
        Environment.GetEnvironmentVariable("Path") + ";" + path)
    )    

#I "./packages/CNTK.CPUOnly/lib/net45/x64/"
#I "./packages/CNTK.CPUOnly/support/x64/Dependency/"
#I "./packages/CNTK.CPUOnly/support/x64/Dependency/Release/"
#I "./packages/CNTK.CPUOnly/support/x64/Release/"

#r "./packages/CNTK.CPUOnly/lib/net45/x64/Cntk.Core.Managed-2.3.1.dll"
open CNTK

// utilities

let shape (dims:int seq) = NDShape.CreateNDShape dims

let isSweepEnd (minibatchValues: seq<MinibatchData>) =
    minibatchValues 
    |> Seq.exists(fun a -> a.sweepEnd)

type Initializer = 
    | Value of float
    | GlorotUniform
    | Custom of CNTK.CNTKDictionary

type Param() =

    static member init (dims: int seq, dataType: DataType, init: Initializer) =
        fun (device:DeviceDescriptor) ->
            match init with
            | Value(x) -> new Parameter(shape dims, dataType, x)
            | GlorotUniform -> new Parameter(shape dims, dataType, CNTKLib.GlorotUniformInitializer())
            | Custom(f) -> new Parameter(shape dims, dataType, f)

type DataSource = {
    SourcePath: string
    Streams: (string * int) seq
    }

type StreamProcessing = 
    | FullDataSweep
    | InfinitelyRepeat
let textSource (data:DataSource) =
    let streams = 
        data.Streams
        |> Seq.map (fun (name, dim) -> 
            new StreamConfiguration(name, dim))
        |> ResizeArray
    fun (processing: StreamProcessing) ->
        MinibatchSource.TextFormatMinibatchSource(
            data.SourcePath, 
            streams, 
            match processing with
            | FullDataSweep -> MinibatchSource.FullDataSweep
            | InfinitelyRepeat -> MinibatchSource.InfinitelyRepeat)
            
// Sequential model

type Computation = Variable -> DeviceDescriptor -> Function
type ComputationWithoutInput = DeviceDescriptor -> Function
type Computation2 = Variable -> DeviceDescriptor -> Function * Function
type Computation2WithoutInput = DeviceDescriptor -> Function * Function
type ComputationVariable = Variable -> DeviceDescriptor -> Variable
type ComputationVariableWithoutInput = DeviceDescriptor -> Variable

type Loss = 
    | CrossEntropyWithSoftmax
    | ClassificationError
    | SquaredError

let evaluation (loss:Loss) (predicted:Function, actual:Variable) =
    match loss with
    | CrossEntropyWithSoftmax -> 
        CNTKLib.CrossEntropyWithSoftmax(new Variable(predicted),actual)
    | ClassificationError -> 
        CNTKLib.ClassificationError(new Variable(predicted),actual)
    | SquaredError -> 
        CNTKLib.SquaredError(new Variable(predicted),actual)

type LearningType =
    | SGDLearner
    | MomentumSGDLearner

type Specification = {
    Features: Variable
    Labels: Variable
    Model: Computation
    Loss: Loss
    Eval: Loss
    LearningType: LearningType
    }

type Schedule = {
    Rate: float
    MinibatchSize: int
    }

type Config = {
    MinibatchSize: int
    Epochs: int
    Device: DeviceDescriptor
    Schedule: Schedule
    } 

let learning (predictor:Function) (schedule:Schedule) =   
    let learningRatePerSample = 
        new TrainingParameterScheduleDouble(schedule.Rate, uint32 schedule.MinibatchSize)
    let parameterLearners =
        ResizeArray<Learner>(
            [ 
                Learner.SGDLearner(predictor.Parameters(), learningRatePerSample) 
            ])
    parameterLearners

let learningWithMomentum (predictor:Function) (schedule:Schedule) = 
    let learningRatePerSample = 
        new TrainingParameterScheduleDouble(schedule.Rate, uint32 schedule.MinibatchSize)
    let momentumTimeConstant = CNTKLib.MomentumAsTimeConstantSchedule(256.)    
    let parameterLearners =
        ResizeArray<Learner>(
            [
                Learner.MomentumSGDLearner(
                    predictor.Parameters(),
                    learningRatePerSample,
                    momentumTimeConstant,
                    true) // unitGainMomentum
            ])
    parameterLearners

type TrainingMiniBatchSummary = {
    Loss:float
    Evaluation:float
    Samples:uint32
    TotalSamples:uint32
    }

let minibatchSummary (trainer:Trainer) =
    if trainer.PreviousMinibatchSampleCount () <> (uint32 0)
    then
        {
            Loss = trainer.PreviousMinibatchLossAverage ()
            Evaluation = trainer.PreviousMinibatchEvaluationAverage ()
            Samples = trainer.PreviousMinibatchSampleCount ()
            TotalSamples = trainer.TotalNumberOfSamplesSeen ()
        }
    else
        {
            Loss = Double.NaN
            Evaluation = Double.NaN
            Samples = trainer.PreviousMinibatchSampleCount ()
            TotalSamples = trainer.TotalNumberOfSamplesSeen ()
        }

let basicMinibatchSummary (summary:TrainingMiniBatchSummary) =
    printfn "Total: %-8i Batch: %3i Loss: %.3f Eval: %.3f"
        summary.TotalSamples
        summary.Samples
        summary.Loss
        summary.Evaluation

type Learner () =

    let progress = new Event<TrainingMiniBatchSummary> ()
    member this.MinibatchProgress = progress.Publish

    member this.learn 
        (source:MinibatchSource) 
        (featureStreamName:string, labelsStreamName:string) 
        (config:Config) 
        (spec:Specification) =

        let device = config.Device

        let predictor = spec.Model spec.Features device
        let loss = evaluation spec.Loss (predictor,spec.Labels)
        let eval = evaluation spec.Eval (predictor,spec.Labels)   

        let parameterLearners = 
            match spec.LearningType with
            | SGDLearner -> learning predictor config.Schedule     
            | MomentumSGDLearner -> learningWithMomentum predictor config.Schedule

        let trainer = Trainer.CreateTrainer(predictor, loss, eval, parameterLearners)
        
        let input = spec.Features
        let labels = spec.Labels
        let featureStreamInfo = source.StreamInfo(featureStreamName)
        let labelStreamInfo = source.StreamInfo(labelsStreamName)
        let minibatchSize = uint32 (config.MinibatchSize)

        let rec learnEpoch (step,epoch) = 

            minibatchSummary trainer
            |> progress.Trigger

            if epoch <= 0
            // we are done : return function
            then predictor
            else
                let step = step + 1
                let minibatchData = source.GetNextMinibatch(minibatchSize, device)

                let arguments : IDictionary<Variable, MinibatchData> =
                    [
                        input, minibatchData.[featureStreamInfo]
                        labels, minibatchData.[labelStreamInfo]
                    ]
                    |> dict

                trainer.TrainMinibatch(arguments, device) |> ignore
                
                // MinibatchSource is created with MinibatchSource.InfinitelyRepeat.
                // Batching will not end. Each time minibatchSource completes an sweep (epoch),
                // the last minibatch data will be marked as end of a sweep. We use this flag
                // to count number of epochs.
                let epoch = 
                    if isSweepEnd (minibatchData.Values)
                    then epoch - 1
                    else epoch

                learnEpoch (step, epoch)

        learnEpoch (0, config.Epochs)


[<RequireQualifiedAccess>]
module Layer = 

    type ComputationBuilder() =
        member this.Bind(m:Computation, f:Variable -> Computation) : Computation =
            fun input -> fun device -> f (new Variable(m input device)) input device

        member this.Bind(m:ComputationWithoutInput, f:Variable -> Computation) : Computation =
            fun input -> fun device -> f (new Variable(m device)) input device

        member this.Bind(m:Computation2, f:Variable * Variable -> Computation) : Computation =
            fun input -> fun device -> let func1, func2 = m input device in f (new Variable(func1), new Variable(func2)) input device

        member this.Bind(m:Computation2WithoutInput, f:Variable * Variable -> Computation) : Computation =
            fun input -> fun device -> let func1, func2 = m device in f (new Variable(func1), new Variable(func2)) input device  

        member this.Bind(m:Function, f:Variable -> Computation) : Computation =
            fun input -> fun device -> f (new Variable(m)) input device        

        member this.Return(x:Variable) : Computation = fun input -> fun device -> x.ToFunction()

    let computation = ComputationBuilder()

    type Computation2Builder() =

        member this.Bind(m:Computation2, f:Variable * Variable -> Computation2) : Computation2 =
            fun input -> fun device -> let func1, func2 = m input device in f (new Variable(func1), new Variable(func2)) input device

        member this.Bind(m:Computation2WithoutInput, f:Variable * Variable -> Computation2) : Computation2 =
            fun input -> fun device -> let func1, func2 = m device in f (new Variable(func1), new Variable(func2)) input device

        member this.Bind(m:ComputationVariable, f:Variable -> Computation2) : Computation2 =
            fun input -> fun device -> f (m input device) input device

        member this.Bind(m:ComputationVariableWithoutInput, f:Variable -> Computation2) : Computation2 =
            fun input -> fun device -> f (m device) input device       

        member this.Bind(m:Function, f:Variable -> Computation2) : Computation2 =
            fun input -> fun device -> f (new Variable(m)) input device        

        member this.Return(x:Variable*Variable) : Computation2 = fun input -> fun device -> ((fst x).ToFunction(), (snd x).ToFunction())

    let computation2 = Computation2Builder()
    
    // Combine 2 Computation Layers into 1
    let stack (next:Computation) (curr:Computation) : Computation =
        computation {
            let! intermediate = curr
            let! result = next intermediate
            return result 
        }

    // combine a sequence of Computation Layers into 1
    let sequence (computations: Computation seq) =
        computations
        |> Seq.reduce (fun acc c -> stack c acc)

    let scale<'T> (scalar:'T) : Computation = 
        fun input ->
            fun device ->
                CNTKLib.ElementTimes(Constant.Scalar<'T>(scalar, device), input)

    let dense (outputDim:int) : Computation =
        fun input ->
            fun device ->
               
                let input : Variable =
                    if (input.Shape.Rank <> 1)
                    then
                        let newDim = input.Shape.Dimensions |> Seq.reduce (*)
                        new Variable(CNTKLib.Reshape(input, shape [ newDim ]))
                    else input

                let inputDim = input.Shape.[0]
                let dataType = input.DataType

                let weights = 
                    new Parameter(
                        shape [outputDim; inputDim], 
                        dataType,
                        CNTKLib.GlorotUniformInitializer(
                            float CNTKLib.DefaultParamInitScale,
                            CNTKLib.SentinelValueForInferParamInitRank,
                            CNTKLib.SentinelValueForInferParamInitRank, 
                            uint32 1),
                        device, 
                        "weights")

                let product = 
                    new Variable(CNTKLib.Times(weights, input, "product"))

                let bias = new Parameter(shape [ outputDim ], 0.0f, device, "bias")
                CNTKLib.Plus(bias, product)

    let dropout (proba:float) : Computation = 
        fun input ->
            fun device ->
                CNTKLib.Dropout(input,proba)     

    let embedding(embeddingDim:int) : Computation =
        fun input ->
            fun device ->
                let inputDim = input.Shape.[0]
                let dataType = input.DataType
                let embeddingParameters = 
                    new Parameter(
                        shape [ embeddingDim; inputDim ], 
                        dataType, 
                        CNTKLib.GlorotUniformInitializer(), 
                        device
                        )
                CNTKLib.Times(embeddingParameters, input)

    let stabilize<'ElementType> : Computation =
        fun input ->
            fun device ->
                let isFloatType = (typeof<'ElementType> = typeof<System.Single>)
                
                let f, fInv =
                    if (isFloatType)
                    then
                        Constant.Scalar(4.0f, device),
                        Constant.Scalar(DataType.Float,  1.0 / 4.0) 
                    else
                        Constant.Scalar(4.0, device),
                        Constant.Scalar(DataType.Double, 1.0 / 4.0)
                    
                let beta = 
                    CNTKLib.ElementTimes(
                        fInv,
                        new Variable(
                            CNTKLib.Log(
                                new Variable(
                                    Constant.Scalar(f.DataType, 1.0) +  
                                    new Variable(
                                        CNTKLib.Exp(
                                            new Variable(
                                                CNTKLib.ElementTimes(f, new Parameter(new NDShape(), f.DataType, 0.99537863, device))
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                
                CNTKLib.ElementTimes(new Variable(beta), input)

    let LSTMPCellWithSelfStabilization<'ElementType> (prevOutput:Variable) (prevCellState:Variable) : Computation2 =
        fun input ->
            fun device ->
                let outputDim = prevOutput.Shape.[0]
                let cellDim = prevCellState.Shape.[0]
                
                let isFloatType = (typeof<'ElementType> = typeof<System.Single>)

                let dataType : DataType = 
                    if isFloatType 
                    then DataType.Float 
                    else DataType.Double

                let createBiasParam : int -> Parameter =
                    match dataType with
                    | DataType.Float -> fun dim -> new Parameter(shape [ dim ], 0.01f, device, "")
                    | DataType.Double -> fun dim -> new Parameter(shape [ dim ], 0.01, device, "")
                    | _ -> failwith "not implemented"

                // TODO: replace by a function...
                let seeder =
                    let mutable s = uint32 1
                    fun () ->
                        s <- s + uint32 1
                        s

                let createProjectionParam : int -> Parameter = 
                    fun oDim -> 
                        new Parameter(
                            shape [ oDim; NDShape.InferredDimension ],
                            dataType, 
                            CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seeder ()), 
                            device
                            )
                
                let createDiagWeightParam : int -> Parameter = 
                    fun dim ->
                        new Parameter(
                            shape [ dim ], 
                            dataType, 
                            CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seeder ()), 
                            device
                            )

                let stabilizedPrevOutput : Function = stabilize<'ElementType> prevOutput device
                let stabilizedPrevCellState : Function = stabilize<'ElementType> prevCellState device

                let projectInput : unit -> Variable = 
                    fun () -> new Variable(createBiasParam (cellDim) + new Variable(createProjectionParam(cellDim) * input))

                // holy lambdas this is nasty
                // Input gate
                let it : Function =
                    CNTKLib.Sigmoid(
                        new Variable(
                            new Variable(projectInput () + new Variable((createProjectionParam (cellDim) * new Variable(stabilizedPrevOutput)))) 
                            + new Variable(CNTKLib.ElementTimes(createDiagWeightParam (cellDim), new Variable(stabilizedPrevCellState)))
                            )
                        )
                        
                let bit : Function = 
                    CNTKLib.ElementTimes(
                        new Variable(it),
                        new Variable(
                            CNTKLib.Tanh(
                                new Variable(projectInput () + new Variable(createProjectionParam(cellDim) * new Variable(stabilizedPrevOutput)))
                                )
                            )
                        )

                // Forget-me-not gate
                let ft : Function = 
                    CNTKLib.Sigmoid(
                        new Variable(
                            projectInput () + 
                            new Variable(
                                new Variable(createProjectionParam(cellDim) * new Variable(stabilizedPrevOutput)) +
                                new Variable(
                                    CNTKLib.ElementTimes(createDiagWeightParam(cellDim), new Variable(stabilizedPrevCellState))
                                    )
                                )
                            )
                        )
                            
                let bft : Function = CNTKLib.ElementTimes(new Variable(ft), prevCellState)

                let ct : Function = new Variable(bft) + new Variable(bit)

                // Output gate
                let ot : Function = 
                    CNTKLib.Sigmoid(
                        new Variable(
                            new Variable (
                                projectInput () + new Variable(createProjectionParam(cellDim) * new Variable(stabilizedPrevOutput))) + 
                            new Variable (
                                CNTKLib.ElementTimes(
                                    createDiagWeightParam(cellDim), 
                                    new Variable(stabilize<'ElementType> (new Variable(ct)) device)
                                    )
                                )
                            )
                        )
                
                let ht : Function = CNTKLib.ElementTimes(new Variable(ot), new Variable(CNTKLib.Tanh(new Variable(ct))))

                let c : Function = ct
                let h : Function = 
                    if (outputDim <> cellDim) 
                    then (createProjectionParam(outputDim) * new Variable(stabilize<'ElementType> (new Variable(ht)) device))
                    else ht

                (h, c)            
            
    let placeholderVariable (shape:NDShape) : ComputationVariable =
        fun input ->
            fun device ->
                Variable.PlaceholderVariable(shape, input.DynamicAxes)

    let LSTMPComponentWithSelfStabilization<'ElementType> (outputShape:NDShape) (cellShape:NDShape) (recurrenceHookH:Variable -> Function) (recurrenceHookC:Variable -> Function) : Computation2 =
        computation2 {
            let! dh = placeholderVariable outputShape
            let! dc = placeholderVariable cellShape

            let! LSTMCell = LSTMPCellWithSelfStabilization<'ElementType> dh dc
            let! actualDh = recurrenceHookH (fst LSTMCell)
            let! actualDc = recurrenceHookC (snd LSTMCell)

            // TODO check this, seems to involve some mutation
            // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
            let replacement : IDictionary<Variable, Variable> =
                [
                    dh, actualDh
                    dc, actualDc
                ]
                |> dict

            (fst LSTMCell).ToFunction().ReplacePlaceholders(replacement) |> ignore

            return LSTMCell 
        }

    /// <summary>
    /// Build a one direction recurrent neural network (RNN) with long-short-term-memory (LSTM) cells.
    /// http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    /// </summary>
    /// <param name="input">the input variable</param>
    /// <param name="numOutputClasses">number of output classes</param>
    /// <param name="embeddingDim">dimension of the embedding layer</param>
    /// <param name="LSTMDim">LSTM output dimension</param>
    /// <param name="cellDim">cell dimension</param>
    /// <param name="device">CPU or GPU device to run the model</param>
    /// <param name="outputName">name of the model output</param>
    /// <returns>the RNN model</returns>
    let LSTMSequenceClassifierNet (numOutputClasses:int) (embeddingDim:int) (LSTMDim:int) (cellDim:int) =
        computation {
            let! embedding = embedding embeddingDim
            let pastValueRecurrenceHook : (Variable -> Function) = fun x -> CNTKLib.PastValue(x)
            let! LSTM, _ = LSTMPComponentWithSelfStabilization<single> (shape [ LSTMDim ]) (shape [ cellDim ]) pastValueRecurrenceHook pastValueRecurrenceHook embedding
            let! thoughtVector = CNTKLib.SequenceLast(LSTM)
            let! dense = dense numOutputClasses thoughtVector

            return dense
        }

[<RequireQualifiedAccess>]
module Activation = 

    let ReLU : Computation = 
        fun input ->
            fun device ->
                CNTKLib.ReLU(input)

    let sigmoid : Computation = 
        fun input ->
            fun device ->
                CNTKLib.Sigmoid(input)

    let tanh : Computation = 
        fun input ->
            fun device ->
                CNTKLib.Tanh(input)


[<RequireQualifiedAccess>]
module Conv2D = 
    
    type Kernel = {
        Width: int
        Height: int
        }

    type Conv2D = {
        Kernel: Kernel 
        OutputFeatures: int
        Initializer: Initializer
        }

    let conv2D = {
        Kernel = { Width = 1; Height = 1 } 
        OutputFeatures = 1
        Initializer = GlorotUniform
        }

    let convolution (args:Conv2D) : Computation = 
        fun input ->
            fun device ->
                let kernel = args.Kernel
                let inputChannels = input.Shape.Dimensions.[2]
                let convParams = 
                    device
                    |> Param.init (
                        [ kernel.Width; kernel.Height; inputChannels; args.OutputFeatures ], 
                        DataType.Float,
                        args.Initializer)

                CNTKLib.Convolution(
                    convParams, 
                    input, 
                    shape [ 1; 1; inputChannels ]
                    )

    type Window = {
        Width: int
        Height: int          
        }

    type Stride = {
        Horizontal: int
        Vertical: int
        }

    type Pool2D = {
        Window: Window
        Stride : Stride 
        PoolingType : PoolingType
        }                
    let pooling (args:Pool2D) : Computation = 
        fun input ->
            fun device ->

                let window = args.Window
                let stride = args.Stride

                CNTKLib.Pooling(
                    input, 
                    args.PoolingType,
                    shape [ window.Width; window.Height ], 
                    shape [ stride.Horizontal; stride.Vertical ], 
                    [| true |]
                    )
let private dictAdd<'K,'V> (key,value) (dict:Dictionary<'K,'V>) = 
    dict.Add(key,value)
    dict

let dataMap xs = 
    let dict = Dictionary<Variable,Value>()
    xs |> Seq.fold (fun dict (var,value) -> dictAdd (var,value) dict) dict
