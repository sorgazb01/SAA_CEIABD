using ComercialOffer;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

var mlContext = new MLContext(seed: 0);

string dataPath = "Data/customers.csv";

// 1. Cargar datos
var data = mlContext.Data.LoadFromTextFile<CustomerData>
(
    path: dataPath,
    hasHeader: true,
    separatorChar: ','
);

// 2. Dividir en train/test
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
var trainData = split.TrainSet;
var testData = split.TestSet;

// 3. Pipeline común de preprocesamiento
var preprocessingPipelineCommon = mlContext.Transforms.Categorical.OneHotEncoding("RegionEncoded", "Region")
.Append(mlContext.Transforms.Conversion.ConvertType("SubcribedNewsletterFloat", "SubcribedNewsletter", Microsoft.ML.Data.DataKind.Single))
.Append(mlContext.Transforms.Concatenate("Features", "Age", "Income", "PreviousPurchases", "WebVisits", "SubcribedNewsletterFloat", "RegionEncoded"));

// 4. Modelo 1: FastForest (bagging)
var fastForestOptions = new FastForestBinaryTrainer.Options
{
    NumberOfTrees = 50,
    NumberOfLeaves = 20,
    MinimumExampleCountPerLeaf = 10,
    FeatureFraction = 0.8,
    FeatureFirstUsePenalty = 0.1,
};

var fastForestPipeline = preprocessingPipelineCommon.Append(mlContext.BinaryClassification.Trainers.FastForest(fastForestOptions));
var fastForestModel = fastForestPipeline.Fit(trainData);
var fastForestPredictions = fastForestModel.Transform(testData);
var fastForestMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(fastForestPredictions, labelColumnName: "Label");

// 5. Modelo 2: LightGbm (gradient boosting)
var lightGbmOptions = new LightGbmBinaryTrainer.Options
{
    NumberOfIterations = 200,
    NumberOfLeaves = 10,
    MinimumExampleCountPerLeaf = 25,
    LearningRate = 0.05f,
};

var lightGbmPipeline = preprocessingPipelineCommon.Append(mlContext.BinaryClassification.Trainers.LightGbm(lightGbmOptions));
var lightGbmModel = lightGbmPipeline.Fit(trainData);
var lightGbmPredictions = lightGbmModel.Transform(testData);
var lightGbmMetrics = mlContext.BinaryClassification.Evaluate(lightGbmPredictions, labelColumnName: "Label");

// 6. Mostrar resultados
Console.WriteLine("===== RESULTADOS FASTFOREST =====");
PrintFastForestMetrics(fastForestMetrics);

Console.WriteLine();
Console.WriteLine("===== RESULTADOS LIGHTGBM =====");
PrintLightGbmMetrics(lightGbmMetrics);

// 7. Comparación
Console.WriteLine();
Console.WriteLine("===== COMPARACIÓN FINAL =====");

string mejorAccuracy =
    fastForestMetrics.Accuracy > lightGbmMetrics.Accuracy
    ? "FastForest"
    : "LightGbm";

string mejorF1 =
    fastForestMetrics.F1Score > lightGbmMetrics.F1Score
    ? "FastForest"
    : "LightGbm";

string mejorAuc =
    fastForestMetrics.AreaUnderRocCurve > lightGbmMetrics.AreaUnderRocCurve
    ? "FastForest"
    : "LightGbm";

Console.WriteLine($"Mejor Accuracy: {mejorAccuracy}");
Console.WriteLine($"Mejor F1 Score: {mejorF1}");
Console.WriteLine($"Mejor AUC: {mejorAuc}");

static void PrintFastForestMetrics(BinaryClassificationMetrics metrics)
{
    Console.WriteLine($"Accuracy: {metrics.Accuracy:F4}");
    Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
    Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");
    Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:F4}");
    Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F4}");
    Console.WriteLine("Matriz de confusión:");
    Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
}

static void PrintLightGbmMetrics(CalibratedBinaryClassificationMetrics metrics)
{
    Console.WriteLine($"Accuracy: {metrics.Accuracy:F4}");
    Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
    Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");
    Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:F4}");
    Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F4}");
    Console.WriteLine($"LogLoss: {metrics.LogLoss:F4}");
    Console.WriteLine("Matriz de confusión:");
    Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
}