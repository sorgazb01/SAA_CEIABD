using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

const string fileInputPath = "clientes_casarural.csv";

MLContext mlContext = new();

IDataView data = mlContext.Data.LoadFromTextFile<ClienteInput>(path: fileInputPath, hasHeader: true, separatorChar: ',');

var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

for (int k = 2; k <= 8; k++)
{
    Console.WriteLine("K = " + k);

    var pipeline = mlContext.Transforms.Concatenate(
            outputColumnName: "Features", 
            nameof(ClienteInput.Edad),
            nameof(ClienteInput.NochesPorEstancia),
            nameof(ClienteInput.ViajaConNinos),
            nameof(ClienteInput.GastoMedio),
            nameof(ClienteInput.DistanciaKm),
            nameof(ClienteInput.ReservasUltimoAnio))
        .Append(mlContext.Clustering.Trainers.KMeans(numberOfClusters: k));

    var model = pipeline.Fit(splitData.TrainSet);
    var predictions = model.Transform(splitData.TestSet);

    ClusteringMetrics metrics = mlContext.Clustering.Evaluate(
        data: predictions, 
        scoreColumnName: "Score", 
        featureColumnName: "Features");
    Console.WriteLine($"Average Distance: {metrics.AverageDistance:F4}");
    Console.WriteLine($"Davies Bouldin Index: {metrics.DaviesBouldinIndex:F4}");
    Console.WriteLine($"Normalized Mutual Information: {metrics.NormalizedMutualInformation:F4}");

    var engine = mlContext.Model.CreatePredictionEngine<ClienteInput, ClusterPrediction>(model);

    var clientes = mlContext.Data.CreateEnumerable<ClienteInput>(data, reuseRowObject: false).ToList();
    
    var resultados = new List<(ClienteInput Cliente, uint ClusterId)>();
    
    foreach (var c in clientes)
    {
        var pred = engine.Predict(c);
        resultados.Add((c, pred.PredictedLabel));
    }
    
    foreach (var grp in resultados.GroupBy(r => r.ClusterId).OrderBy(g => g.Key))
    {
        Console.WriteLine($"\nCluster {grp.Key}");
        
        Console.WriteLine($"   Edad media: {grp.Average(r => r.Cliente.Edad):F1}");
        Console.WriteLine($"   Noches por estancia: {grp.Average(r => r.Cliente.NochesPorEstancia):F1}");
        Console.WriteLine($"   % que viaja con niños: {grp.Average(r => r.Cliente.ViajaConNinos) * 100:F1}%");
        Console.WriteLine($"   Gasto medio: {grp.Average(r => r.Cliente.GastoMedio):F0} €");
        Console.WriteLine($"   Distancia media: {grp.Average(r => r.Cliente.DistanciaKm):F0} km");
        Console.WriteLine($"   Reservas último año: {grp.Average(r => r.Cliente.ReservasUltimoAnio):F1}");
    }
}

public class ClienteInput
{
    [LoadColumn(0)]
    public float IdCliente { get; set; }
    
    [LoadColumn(1)]
    public float Edad { get; set; }
    
    [LoadColumn(2)]
    public float NochesPorEstancia { get; set; }
    
    [LoadColumn(3)]
    public float ViajaConNinos { get; set; }
    
    [LoadColumn(4)]
    public float GastoMedio { get; set; }
    
    [LoadColumn(5)]
    public float DistanciaKm { get; set; }
    
    [LoadColumn(6)]
    public float ReservasUltimoAnio { get; set; }
}

public class ClusterPrediction
{
    [ColumnName("PredictedLabel")]
    public uint PredictedLabel { get; set; }

    [ColumnName("Score")]
    public float[]? Distances { get; set; }
}
