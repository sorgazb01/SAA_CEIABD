using System.Data;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

const string fileInputPath = "puntos-100.csv";

MLContext mlContext = new();

IDataView data = mlContext.Data.LoadFromTextFile<Point>(path: fileInputPath, hasHeader: true, separatorChar: ',');
var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

for (int i = 2; i <= 8; i++)
{

    Console.WriteLine($"\n\nNumber of clusters: {i}");

    var pipeline = mlContext.Transforms.Concatenate(outputColumnName: "Features", ["x", "y",])
    .Append(mlContext.Clustering.Trainers.KMeans(numberOfClusters: i));

    var model = pipeline.Fit(splitData.TrainSet);
    var predictions = model.Transform(splitData.TestSet);

    ClusteringMetrics metrics = mlContext.Clustering.Evaluate(data: predictions, scoreColumnName: "Score", featureColumnName: "Features");

    Console.WriteLine($"Average Distance: {metrics.AverageDistance:F4}");
    Console.WriteLine($"Davies Bouldin Index: {metrics.DaviesBouldinIndex:F4}");
    Console.WriteLine($"Normalized Mutual Information: {metrics.NormalizedMutualInformation:F4}");

    var engine = mlContext.Model.CreatePredictionEngine<Point, PointPrediction>(model);

    Point [] pointsToPredict = [

        new Point() { x = 1, y = 1 },
        new Point() { x = 5, y = 5 },
        new Point() { x = 10, y = 10 },
        new Point() { x = 10, y = 1 },
    ];

    PointPrediction predictedPoint;
}

public class Point
{
    [LoadColumn(1)]
    public float x { get; set; }
    [LoadColumn(2)]
    public float y { get; set; }
}

public class PointPrediction
{
    [ColumnName("PredictedLabel")]
    public uint ClusterId { get; set; }

    [ColumnName("Score")]
    public float[]? Distances { get; set; }
}