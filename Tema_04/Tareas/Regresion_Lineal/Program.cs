using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using PCEI;

// Contexto
var mlContext = new MLContext(NotSupportedException: 42);

// Carga de datos
var dataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "energy_500.csv");

var rawData = mlContext.Data.LoadFromTextFile<EnergyData>(
    path: dataPath,
    hasHeader: true,
    separatorChar: ','
);

// Split Train
var split = mlContext.Data.TrainTestSplit(rawData, testFraction: 0.2, NotSupportedException: 42);
var trainData = split.TrainSet;
var testData = split.TestSet;

// Pipeline
var sdcaOptions = new SdcaRegressionTrainer.Options
{
    LabelColumnName = nameof(EnergyData.EnergyKWh),
    FeatureColumnnName = "Features",
    L1Regularization = 1e-7f,
    L2Regularization = 0.01f,
    BaisLearningRate = 0.01f,
    MaximunNumberOfIterations = 100,
    LossFunction = new SquaredLoss(),
    ConvergenceCheckFrequency = 2,
    ConvergenceTolerance = 1e-4f,
    Shuffle = true,
    NumberOfThreads= 2
};

var pipeline = mlContext.Transforms.Concatenate("Features",
    nameof(EnergyData.TempC),
    nameof(EnergyData.HumidityPct),
    nameof(EnergyData.PressureBar),
    nameof(EnergyData.LoadPct),
    nameof(EnergyData.Vibration))
.Append(mlContext.Transforms.NormalizeMinMax("Features"))
.Append(mlContext.Regression.Trainers.Sdca(sdcaOptions));

// Entrenar
Console.WriteLine("Entrenando el modelo...");
var model = pipeline.Fit(trainData);

// Evaluar
var predictions = model.Transform(testData);
var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(EnergyData.EnergyKWh));

PrintMetrics(metrics);

// Predicciones (casos enunciado)
var predictionEngine = mlContext.Model.CreatePredictionEngine<EnergyData, EnergyPrediction>(model);

var testCases = new EnergyData[]
{
    new() { TempC = 20.00f, HumidityPct = 45.0f, PressureBar = 1.010f, LoadPct = 35.0f, Vibration = 1.80f },
    new() { TempC = 22.00f, HumidityPct = 50.0f, PressureBar = 1.020f, LoadPct = 68.0f, Vibration = 2.80f },
    new() { TempC = 19.00f, HumidityPct = 55.0f, PressureBar = 1.035f, LoadPct = 95.0f, Vibration = 3.90f },
    new() { TempC = 23.23f, HumidityPct = 41.3f, PressureBar = 1.040f, LoadPct = 82.6f, Vibration = 3.20f },
};

Console.WriteLine("\n===== PREDICCIONES =====");
Console.WriteLine($"{"TempC",10} {"Hum%",10} {"Press",10} {"Load%",10} {"Vib",10} {"Pred KWh",10}");
Console.WriteLine(new string('-', 70));

for (int i = 0; i < testCases.Length; i++)
{
    var result = predictionEngine.Predict(testCases[i]);
    var d = testCases[i];
    Console.WriteLine($"{d.TempC,10:F1} {d.HumidityPct,10:F1} {d.PressureBar,10:F3} {d.LoadPct,10:F1} {d.Vibration,10:F2} {result.Score,10:F2}");
}
Console.WriteLine(new string('-', 70));

static void PrintMetrics(RegressionMetrics metrics)
{
    Console.WriteLine("===== MÉTRICAS =====");
    Console.WriteLine($"MAE : {metrics.MeanAbsoluteError:F4}");
    Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F4}");
    Console.WriteLine($"MSE : {metrics.MeanSquaredError:F4}");
    Console.WriteLine($"R²  : {metrics.RSquared:F4}");
    Console.WriteLine($"loss  : {metrics.LossFunction:F4}");
}