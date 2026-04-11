using Microsoft.ML;

// Datos de entrenamiento
var trainingPoints = new List<SvmDataPoint>
{
    new() { Features = [1.0f, 1.0f], Label = true },
    new() { Features = [1.5f, 2.0f], Label = true },
    new() { Features = [1.2f, 0.8f], Label = true },
    new() { Features = [0.9f, 1.1f], Label = true },
    new() { Features = [1.3f, 1.4f], Label = true },
    new() { Features = [1.7f, 1.6f], Label = true },
    new() { Features = [0.8f, 0.9f], Label = true },
    new() { Features = [1.4f, 1.0f], Label = true },
    new() { Features = [2.2f, 2.4f], Label = true },

    new() { Features = [3.0f, 3.5f], Label = false },
    new() { Features = [4.0f, 4.5f], Label = false },
    new() { Features = [3.2f, 3.7f], Label = false },
    new() { Features = [3.8f, 4.2f], Label = false },
    new() { Features = [4.1f, 3.9f], Label = false },
    new() { Features = [3.6f, 4.4f], Label = false },
    new() { Features = [2.9f, 3.3f], Label = false },
    new() { Features = [4.3f, 4.1f], Label = false },
    new() { Features = [2.5f, 2.7f], Label = false }
};

// Datos de test
var testPoints = new List<SvmDataPoint>
{
    new() { Features = [1.4f, 1.3f], Label = true },
    new() { Features = [3.5f, 4.0f], Label = false },
    new() { Features = [1.1f, 0.9f], Label = true },
    new() { Features = [3.2f, 3.8f], Label = false },
    new() { Features = [2.3f, 2.5f], Label = true },
    new() { Features = [2.6f, 2.8f], Label = false },
    new() { Features = [1.6f, 1.5f], Label = true },
    new() { Features = [4.2f, 4.0f], Label = false }
};

var mlContext = new MLContext(seed: 0);

IDataView trainData = mlContext.Data.LoadFromEnumerable(trainingPoints);
IDataView testData = mlContext.Data.LoadFromEnumerable(testPoints);

// Definimos el trainer
var trainer = mlContext.BinaryClassification.Trainers.LinearSvm(
    labelColumnName: "Label",
    featureColumnName: "Features",
    numberOfIterations: 100
);

// Entrenamos el modelo
var model = trainer.Fit(trainData);

// Obtenemos las predicciones sobre el conjunto de test
var predictions = model.Transform(testData);

// LinearSvm no está calibrado, se usa EvaluateNonCalibrated
var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(predictions);

Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
Console.WriteLine($"Precision: {metrics.PositivePrecision:F2}");
Console.WriteLine($"Recall: {metrics.PositiveRecall:F2}");
Console.WriteLine($"Neg.Precision: {metrics.NegativePrecision:F2}");
Console.WriteLine($"Neg.Recall: {metrics.NegativeRecall:F2}");

// Predicción de un punto concreto
var predictionEngine = mlContext.Model.CreatePredictionEngine<SvmDataPoint, SvmPrediction>(model);

var ejemplo = new SvmDataPoint { Features = [1.4f, 1.3f] };
var resultado = predictionEngine.Predict(ejemplo);
Console.WriteLine($"La clase predicha es: {(resultado.PredictedLabel ? "Rojo" : "Azul")}");
