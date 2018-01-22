(*
https://github.com/Microsoft/CNTK/master/Examples/TrainingCSharp/Common/LSTMSequenceClassifier.cs
*)


#load "../../CNTK.CE.fsx"

open System.IO
open System.Collections.Generic
open CNTK
open CNTK.CE

let DataFolder = __SOURCE_DIRECTORY__

let inputDim = 2000
let cellDim = 25
let hiddenDim = 25
let embeddingDim = 50
let numOutputClasses = 5

// build the model
let featuresName = "features"
let features = Variable.InputVariable(shape [ inputDim ], DataType.Float, featuresName, null, true)
let labelsName = "labels"
let labels = Variable.InputVariable(shape [ numOutputClasses ], DataType.Float, labelsName, ResizeArray<Axis>( [ Axis.DefaultBatchAxis() ]), true)

let network : Computation =
    Layer.computation {
        let! classifierOutput = Layer.LSTMSequenceClassifierNet numOutputClasses embeddingDim hiddenDim cellDim

        return classifierOutput
    }

// prepare training data
let streamConfigurations = 
    [|
        new StreamConfiguration(featuresName, inputDim, true, "x")
        new StreamConfiguration(labelsName, numOutputClasses, false, "y")
    |]

let minibatchSource = 
    MinibatchSource.TextFormatMinibatchSource(
        Path.Combine(DataFolder, "Train.ctf"), 
        streamConfigurations, 
        MinibatchSource.InfinitelyRepeat, 
        true
        )

let basicMinibatchSummary (summary:TrainingMiniBatchSummary) =
    printfn "Total: %-8i Batch: %3i Loss: %.3f Eval: %.3f"
        summary.TotalSamples
        summary.Samples
        summary.Loss
        summary.Evaluation

let trainer = Learner ()
trainer.MinibatchProgress.Add(basicMinibatchSummary)

// train the model
let minibatchSize = uint32 200
let outputFrequencyInMinibatches = 20
let miniBatchCount = 0
let numEpochs = 5

let config = {
    MinibatchSize = 64
    Epochs = 5
    Device = DeviceDescriptor.CPUDevice
    Schedule = { Rate = 0.0005; MinibatchSize = 1 }
    }

    
let spec = {
    Features = features
    Labels = labels
    Model = network
    Loss = CrossEntropyWithSoftmax
    Eval = ClassificationError
    LearningType = MomentumSGDLearner
    }

let predictor = trainer.learn minibatchSource (featuresName, labelsName) config spec

let modelFile = Path.Combine(__SOURCE_DIRECTORY__,"MNISTConvolution.model")

predictor.Save(modelFile)