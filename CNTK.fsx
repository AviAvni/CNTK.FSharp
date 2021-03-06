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

let private dictAdd<'K,'V> (key,value) (dict:Dictionary<'K,'V>) = 
    dict.Add(key,value)
    dict

let dataMap xs = 
    let dict = Dictionary<Variable,Value>()
    xs |> Seq.fold (fun dict (var,value) -> dictAdd (var,value) dict) dict

type Activation = 
    | None
    | ReLU
    | Sigmoid
    | Tanh


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

let progress (trainer:Trainer, frequency) current =
    if current % frequency = 0
    then Some(current, minibatchSummary trainer)
    else Option.None

let printer = 
    Option.iter (fun (batch,report) ->
        printf "Batch: %i " batch
        printf "Average Loss: %.3f " (report.Loss)
        printf "Average Eval: %.3f " (report.Evaluation)
        printfn ""
        )

module Debug = 

    let debugDevice = DeviceDescriptor.CPUDevice

    let valueAt<'T> (value:'T seq) (f:Function) =

        let input = f.Inputs |> Seq.find (fun arg -> arg.IsInput)
        let inputValue = Value.CreateBatch(input.Shape, value, debugDevice)
        let output = f.Output
        let inputMap = 
            let map = Dictionary<Variable,Value>()
            map.Add(input,inputValue)
            map
        let outputMap =
            let map = Dictionary<Variable,Value>()
            map.Add(output, null)
            map
        f.Evaluate(inputMap,outputMap,debugDevice)
        outputMap.[output].GetDenseData<'T>(output) 
        |> Seq.map (fun x -> x |> Seq.toArray)
        |> Seq.toArray

    // TODO: fix this, not quite working
    let valueOf (name:string) (f:Function) =
        let param = 
            f.Inputs 
            |> Seq.tryFind (fun arg -> arg.Name = name)

        param
        |> Option.map (fun p ->
            let inputMap = Dictionary<Variable,Value>()
            let outputMap =
                let map = Dictionary<Variable,Value>()
                map.Add(p, null)
                map
            f.Evaluate(inputMap,outputMap,debugDevice)
            outputMap.[p].GetDenseData<float>(p) 
            |> Seq.map (fun x -> x |> Seq.toArray)
            |> Seq.toArray
            )
