using Microsoft.ML.Data;

namespace AnomalDetect.Models;

public class HvacData
{
    [LoadColumn(0)]
    public float Minute { get; set; }

    [LoadColumn(1)]
    public float TempC { get; set; }

    [LoadColumn(2)]
    public float HumPct { get; set; }

    [LoadColumn(3)]
    public float PowerW { get; set; }

    [LoadColumn(4)]
    public float DeltaT { get; set; }

    [LoadColumn(5)]
    public float Label { get; set; }
}

public class HvacPrediction : HvacData
{
    // true = anomalía
    public bool PredictedLabel { get; set; }

    // Cuanto mayor, más anómalo
    // Por defecto >= 0.5 anomalía
    public float Score { get; set; }
}