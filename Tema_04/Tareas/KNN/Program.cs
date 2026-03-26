var trainingData = new List<DataPoint>
{
    new([1.0, 1.0], "Rojo"),
    new([1.5, 2.0], "Rojo"),
    new([1.2, 0.8], "Rojo"),
    new([0.9, 1.1], "Rojo"),
    new([1.3, 1.4], "Rojo"),
    new([1.7, 1.6], "Rojo"),
    new([0.8, 0.9], "Rojo"),
    new([1.4, 1.0], "Rojo"),
    new([2.2, 2.4], "Rojo"),

    new([3.0, 3.5], "Azul"),
    new([4.0, 4.5], "Azul"),
    new([3.2, 3.7], "Azul"),
    new([3.8, 4.2], "Azul"),
    new([4.1, 3.9], "Azul"),
    new([3.6, 4.4], "Azul"),
    new([2.9, 3.3], "Azul"),
    new([4.3, 4.1], "Azul"),
    new([2.5, 2.7], "Azul")
};

var testData = new List<DataPoint>
{
    new([1.4, 1.3], "Rojo"),
    new([3.5, 4.0], "Azul"),
    new([1.1, 0.9], "Rojo"),
    new([3.2, 3.8], "Azul"),
    new([2.3, 2.5], "Rojo"),
    new([2.6, 2.8], "Azul"),
    new([1.6, 1.5], "Rojo"),
    new([4.2, 4.0], "Azul")
};

var knn = new KNN(trainingData);

int k = 4;
double[] nuevoPunto = [1.4, 1.3];

string prediccion = knn.Predict(nuevoPunto, k);
Console.WriteLine($"La clase predicha es: {prediccion}");

var metrics = Metrics.Evaluate(knn, testData, k, "Rojo");

Console.WriteLine($"TP: {metrics.TP}");
Console.WriteLine($"TN: {metrics.TN}");
Console.WriteLine($"FP: {metrics.FP}");
Console.WriteLine($"FN: {metrics.FN}");

Console.WriteLine($"Accuracy: {metrics.Accuracy():F2}");
Console.WriteLine($"Precision: {metrics.Precision():F2}");
Console.WriteLine($"Recall: {metrics.Recall():F2}");
Console.WriteLine($"F1 Score: {metrics.F1():F2}");