using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using AnomalDetect.Models;

////////////////////////////////////
////// Contexto
////////////////////////////////////

var mlContext = new MLContext(seed: 666);

const string DatasetPath = "Data/hvac_anomalies.csv";


////////////////////////////////////
////// Cargamos los datos
////////////////////////////////////

IDataView dataView = mlContext.Data.LoadFromTextFile<HvacData>(
    path: DatasetPath,
    hasHeader: true,
    separatorChar: ',');

// Entreno con los datos "normales", hasta el 30
var trainingData = mlContext.Data.FilterRowsByColumn(
    dataView,
    nameof(HvacData.Minute),
    upperBound: 30);


////////////////////////////////////
////// Pipeline de PROCESAMIENTO
////////////////////////////////////
var preprocessingPipeline =
    mlContext.Transforms.Concatenate("Features",
    [
        // nameof(HvacData.Minute), // No aporta nada
        nameof(HvacData.TempC),
        nameof(HvacData.HumPct),
        nameof(HvacData.PowerW),
        nameof(HvacData.DeltaT),
    ])
    .Append(mlContext.Transforms.NormalizeMeanVariance("Features"));


////////////////////////////////////
////// Pipeline de DETECCIÓN ANOML.
////////////////////////////////////

var rPcaPipeline =
    preprocessingPipeline.Append(
        mlContext.AnomalyDetection.Trainers.RandomizedPca(
            featureColumnName: "Features",
            rank: 2
        )
    );



////////////////////////////////////
////// Creación del modelo
////// M = {PREPRO + RandomicedPCA}
////////////////////////////////////

// semi--supervisado
var model = rPcaPipeline.Fit(trainingData);

// no supervisado
// var model = rPcaPipeline.Fit(dataView);



////////////////////////////////////
////// Modelo anterior con diferentes threshold
////// M = {PREPRO + RandomicedPCA + Custom Threshold}
////////////////////////////////////

foreach (var t in new[] { 0.30f, 0.45f, 0.60f, 0.50f })
{
    var preproModel = model.Take(model.Count() - 1);
    var pca = (AnomalyPredictionTransformer<PcaModelParameters>)model.LastTransformer;

    // Cambia el umbral SOLO del último transformer
    var pcaWithThreshold = mlContext.AnomalyDetection.ChangeModelThreshold(
        pca,
        threshold: t);

    // Reconstruye la cadena sustituyendo el último transformer
    var thresholdedModel = model.Append(pcaWithThreshold);

    // Aplica el modelo con ese umbral
    var scored = thresholdedModel.Transform(dataView);

    var count = mlContext.Data
        .CreateEnumerable<HvacPrediction>(scored, reuseRowObject: false)
        .Count(r => r.PredictedLabel);

    Console.WriteLine($"Threshold {t:F2}: {count} anomalías detectadas");
}

////////////////////////////////////
////// Consumir el modelo
////////////////////////////////////
var predictions = model.Transform(dataView);


////////////////////////////////////
////// Medir el modelo
////////////////////////////////////
var metrics = mlContext.AnomalyDetection.Evaluate(
    data: predictions,
    labelColumnName: "Label",
    scoreColumnName: "Score",
    predictedLabelColumnName: "PredictedLabel",
    falsePositiveCount: 5);

Console.WriteLine($"AU ROC: {metrics.AreaUnderRocCurve:F4}");
Console.WriteLine($"DR at FP=5: {metrics.DetectionRateAtFalsePositiveCount:F4}");

// Creamos un enumerable, para poder visualizar
var results = mlContext.Data.CreateEnumerable<HvacPrediction>(
    predictions,
    reuseRowObject: false);

// Mostramos los datos con un formateo friendly
Console.WriteLine("Min\tTemp\tHum\tPower\tDeltaT\tAnomaly\tScore");
Console.WriteLine("-------------------------------------------------------------");

foreach (var r in results)
{
    Console.WriteLine(
        $"{r.Minute}\t{r.TempC:F1}\t{r.HumPct:F1}\t{r.PowerW:F0}\t{r.DeltaT:F1}\t" +
        $"{(r.PredictedLabel ? "🔴" : "🟢")}\t{r.Score:F4}");
}

Console.WriteLine("-------------------------------------------------------------");
