using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Deteccion_Anomalias.Models;

var mlContext = new MLContext(seed: 666);

const string DataSet = "Data/anomalias_conduccion.csv";

IDataView dataView = mlContext.Data.LoadFromTextFile<ConduccionData>(
    path: DataSet,
    hasHeader: true,
    separatorChar: ','
);

var trainingData = mlContext.Data.FilterRowsByColumn(
    dataView,
    nameof(ConduccionData.Label),
    upperBound: 0.5
);

var preProcessingPipeline = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TipoDeVia_NoCategorica", inputColumnName: nameof(ConduccionData.TipoDeVia))
    .Append(mlContext.Transforms.Concatenate(
        "Features",
        [
            nameof(ConduccionData.Velocidad_km_h),
            nameof(ConduccionData.AceleracionLongitudinal_m_s2),
            nameof(ConduccionData.AceleracionLateral_m_s2),
            nameof(ConduccionData.Jerk_m_s3),
            nameof(ConduccionData.VelocidadGiroYaw_deg_s),
            nameof(ConduccionData.PorcentajeFreno),
            nameof(ConduccionData.PorcentajeAcelerador),
            nameof(ConduccionData.PrecisionGPS_m),
            "TipoDeVia_NoCategorica"
        ]
    )
).Append(mlContext.Transforms.NormalizeMeanVariance("Features"));

var rPcaPipeline = preProcessingPipeline.Append(
    mlContext.AnomalyDetection.Trainers.RandomizedPca(
        featureColumnName: "Features",
        rank: 5
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
        .CreateEnumerable<ConduccionPrediccion>(scored, reuseRowObject: false)
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

var results = mlContext.Data.CreateEnumerable<ConduccionPrediccion>(
    predictions,
    reuseRowObject: false);

Console.WriteLine("Id\tTipoDeVia\tVelocidad_km_h\tAceleracionLongitudinal_m_s2\tAceleracionLateral_m_s2\tJerk_m_s3\tVelocidadGiroYaw_deg_s\tPorcentajeFreno\tPorcentajeAcelerador\tPrecisionGPS_m\tAnomaly\tScore");
Console.WriteLine("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------");

foreach (var r in results)
{
    Console.WriteLine(
        $"{r.Id}\t{r.TipoDeVia}\t{r.Velocidad_km_h:F1}\t{r.AceleracionLongitudinal_m_s2:F1}\t{r.AceleracionLateral_m_s2:F1}\t{r.Jerk_m_s3:F1}\t{r.VelocidadGiroYaw_deg_s:F1}\t{r.PorcentajeFreno:F1}\t{r.PorcentajeAcelerador:F1}\t{r.PrecisionGPS_m:F1}\t{(r.PredictedLabel ? "🔴" : "🟢")}\t{r.Score:F4}");
}

Console.WriteLine("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------");