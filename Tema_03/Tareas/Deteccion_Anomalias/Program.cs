using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Deteccion_Anomalias.Models;

var mlContext = new MLContext(seed: 666);

const string DatasetPath = "Data/manufacturing_anomalies.csv";

IDataView dataView = mlContext.Data.LoadFromTextFile<ManufactoringData>(
    path: DatasetPath,
    hasHeader: true,
    separatorChar: ','
);

var trainingData = mlContext.Data.FilterRowsByColumn(
    dataView,
    nameof(ManufactoringData.Id),
    upperBound: 600
);

var preprocessingPipeline = mlContext.Transforms.Concatenate(
    "Features",
    [
        nameof(ManufactoringData.Weight),
        nameof(ManufactoringData.Length),
        nameof(ManufactoringData.Width),
        nameof(ManufactoringData.Hardness),
        nameof(ManufactoringData.TempProcess),
        nameof(ManufactoringData.VibrationRms)
    ]
).Append(mlContext.Transforms.NormalizeMeanVariance("Features"));

var rPcaPipeline = preprocessingPipeline.Append(
    mlContext.AnomalyDetection.Trainers.RandomizedPca(
        featureColumnName: "Features",
        rank: 2
    )  
);

var model = rPcaPipeline.Fit(trainingData);

foreach (var t in new[] { 0.30f, 0.45f, 0.60f, 0.50f })
{
    var preproModel = model.Take(model.Count() - 1);
    var pca = (AnomalyPredictionTransformer<PcaModelParameters>)model.LastTransformer;

    var pcaWithThreshold = mlContext.AnomalyDetection.ChangeModelThreshold(
        pca,
        threshold: t);

    var thresholdedModel = model.Append(pcaWithThreshold);

    var scored = thresholdedModel.Transform(dataView);

    var count = mlContext.Data
        .CreateEnumerable<ManufactoringPrediction>(scored, reuseRowObject: false)
        .Count(r => r.PredictedLabel);

    Console.WriteLine($"Threshold {t:F2}: {count} anomalías detectadas");
}

var predictions = model.Transform(dataView);


var metrics = mlContext.AnomalyDetection.Evaluate(
    data: predictions,
    labelColumnName: "Label",
    scoreColumnName: "Score",
    predictedLabelColumnName: "PredictedLabel",
    falsePositiveCount: 5);

Console.WriteLine($"AU ROC: {metrics.AreaUnderRocCurve:F4}");
Console.WriteLine($"DR at FP=5: {metrics.DetectionRateAtFalsePositiveCount:F4}");

var results = mlContext.Data.CreateEnumerable<ManufactoringPrediction>(
    predictions,
    reuseRowObject: false);

Console.WriteLine("Id\tWeight\tLength\tWidth\tHardness\tTempProcess\tVibrationRms\tScore");
Console.WriteLine("-------------------------------------------------------------");

foreach (var r in results)
{
    Console.WriteLine(
        $"{r.Id}\t{r.Weight:F1}\t{r.Length:F1}\t{r.Width:F1}\t{r.Hardness:F1}\t{r.TempProcess:F1}\t{r.VibrationRms:F1}\t" +
        $"{(r.PredictedLabel ? "🔴" : "🟢")}\t{r.Score:F4}");
}

Console.WriteLine("-------------------------------------------------------------");