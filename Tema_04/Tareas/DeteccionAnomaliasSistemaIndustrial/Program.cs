using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Calibrators;
using DASI;

var mlContext = new MLContext(seed: 42);

// Ruta del fichero con los datos
var rutaFichero = "Data/system_500.csv";

// Carga de los datos
var rawData = mlContext.Data.LoadFromTextFile<SensorData>(
    path: rutaFichero,
    hasHeader: true,
    separatorChar: ','
);

var split = mlContext.Data.TrainTestSplit(rawData, testFraction: 0.2, seed: 42);
var trainData = split.TrainSet;
var testData = split.TestSet;

// Pipeline
var sdcaOptions = new SdcaLogisticRegressionBinaryTrainer.Options
{
    LabelColumnName = nameof(SensorData.IsAnomaly),
    FeatureColumnName = "Features",
    ConvergenceTolerance = 0.05f,
    MaximumNumberOfIterations = 30,
    PositiveInstanceWeight = 1.2f, 
    L1Regularization = 1e-7f,
    L2Regularization = 0.01f,
    BiasLearningRate = 0.1f,
    Shuffle = true,
    NumberOfThreads = 2,
};

// Pipeline sin normalizacion
var pipelineSinNorm = mlContext.Transforms.Concatenate("Features",
    nameof(SensorData.TempC),
    nameof(SensorData.HumPct),
    nameof(SensorData.PowerW),
    nameof(SensorData.DeltaT))
.Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(sdcaOptions));

// Entrenar el modelo sin normalizacion
var modelSinNorm = pipelineSinNorm.Fit(trainData);

// Evaluar el modelo sin normalizacion
var predictionsSinNorm   = modelSinNorm.Transform(testData);
var metricsSinNorm = mlContext.BinaryClassification.Evaluate(predictionsSinNorm, labelColumnName: nameof(SensorData.IsAnomaly));
Console.WriteLine("===== MODELO SIN NORMALIZACIÓN =====");
PrintLogisticRegressionMetrics(metricsSinNorm);

// Pesos
ShowBiasAndWeights(modelSinNorm.LastTransformer.Model);

// Pipeline con normalizacion
var pipelineNormalizado = mlContext.Transforms.Concatenate("Features",
    nameof(SensorData.TempC),
    nameof(SensorData.HumPct),
    nameof(SensorData.PowerW),
    nameof(SensorData.DeltaT))
.Append(mlContext.Transforms.NormalizeMinMax("Features"))
.Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(sdcaOptions));

// Entrenar el modelo con normalizacion
var modelNormalizado = pipelineNormalizado.Fit(trainData);

// Evaluar el modelo normalizado
var predictionsNormalizado   = modelNormalizado.Transform(testData);
var metricsNormalizado = mlContext.BinaryClassification.Evaluate(predictionsNormalizado, labelColumnName: nameof(SensorData.IsAnomaly));
Console.WriteLine("===== MODELO CON NORMALIZACIÓN =====");
PrintLogisticRegressionMetrics(metricsNormalizado);

// Pesos
ShowBiasAndWeights(modelNormalizado.LastTransformer.Model);

// Predicciones 
var predictionEngine = mlContext.Model.CreatePredictionEngine<SensorData, AnomalyPrediction>(modelNormalizado);

var testCases = new SensorData[]
{
    new() { TempC = 23.0f, HumPct = 40.0f, PowerW = 510.0f, DeltaT = 5.0f  },
    new() { TempC = 37.5f, HumPct = 31.0f, PowerW = 770.0f, DeltaT = 17.5f },
    new() { TempC = 24.5f, HumPct = 39.0f, PowerW = 525.0f, DeltaT = 6.5f  },
    new() { TempC = 40.0f, HumPct = 30.0f, PowerW = 800.0f, DeltaT = 20.0f },
};

Console.WriteLine("\n===== PREDICCIONES =====");
Console.WriteLine($"{"TempC",10} {"Hum%",10} {"PowerW",10} {"DeltaT",10} {"Clase",10} {"Prob.",10}");
Console.WriteLine(new string('-', 65));

for (int i = 0; i < testCases.Length; i++)
{
    var result = predictionEngine.Predict(testCases[i]);
    var d = testCases[i];

    string clase = result.PredictedLabel ? "Anómalo" : "Normal";

    Console.WriteLine($"{d.TempC,10:F1} {d.HumPct,10:F1} {d.PowerW,10:F1} {d.DeltaT,10:F1} {clase,10} {result.Probability,10:P2}");
}

Console.WriteLine(new string('-', 65));

static void PrintLogisticRegressionMetrics(CalibratedBinaryClassificationMetrics metrics)
{
    Console.WriteLine("===== MÉTRICAS =====");
    Console.WriteLine($"Accuracy      : {metrics.Accuracy:F4}");
    Console.WriteLine($"AUC           : {metrics.AreaUnderRocCurve:F4}");
    Console.WriteLine($"F1Score       : {metrics.F1Score:F4}");
    Console.WriteLine($"Precision (N) : {metrics.NegativePrecision:F4}");
    Console.WriteLine($"Recall (N)    : {metrics.NegativeRecall:F4}");
    Console.WriteLine($"Precision (P) : {metrics.PositivePrecision:F4}");
    Console.WriteLine($"Recall (P)    : {metrics.PositiveRecall:F4}");
    Console.WriteLine("----- matrix -----");
    Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
    Console.WriteLine("----- otras -----");
    Console.WriteLine($"loss (log loss)           : {metrics.LogLoss:F4}");
    Console.WriteLine($"loss reduction (log loss) : {metrics.LogLossReduction:F4}");
    Console.WriteLine($"Entropy                   : {metrics.Entropy:F4}");
    Console.WriteLine($"AUC P/R                   : {metrics.AreaUnderPrecisionRecallCurve:F4}");
}

static void SaveLogisticRegressionMetrics(string resultsPath, CalibratedBinaryClassificationMetrics metrics)
{
    var metricsFile = Path.Combine(resultsPath, "metrics.txt");
    File.WriteAllText(metricsFile,
        $"Accuracy      : {metrics.Accuracy:F4}\n"
        + $"AUC           : {metrics.AreaUnderRocCurve:F4}\n"
        + $"F1Score       : {metrics.F1Score:F4}\n"
        + $"Precision (N) : {metrics.NegativePrecision:F4}\n"
        + $"Recall (N)    : {metrics.NegativeRecall:F4}\n"
        + $"Precision (P) : {metrics.PositivePrecision:F4}\n"
        + $"Recall (P)    : {metrics.PositiveRecall:F4}\n"
        + "----- matrix -----\n"
        + $"{metrics.ConfusionMatrix.GetFormattedConfusionTable()}");
}

static void ShowBiasAndWeights(
    CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator> logisticModel)
{
    Console.WriteLine("\n===== SESGO =====");
    Console.WriteLine($"Bias: {logisticModel.SubModel.Bias:F4}");
    Console.WriteLine("\n===== PESOS =====");
    var names   = new[] { "TempC", "HumPct", "PowerW", "DeltaT" };
    var weights = logisticModel.SubModel.Weights.ToArray();
    for (int i = 0; i < weights.Length; i++)
        Console.WriteLine($"{names[i]}: {weights[i]:F4}");
}
