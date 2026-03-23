using Microsoft.ML.Data;

namespace DASI;

public class SensorData
{
    [LoadColumn(0)] public float TempC      { get; set; }
    [LoadColumn(1)] public float HumPct     { get; set; }
    [LoadColumn(2)] public float PowerW     { get; set; }
    [LoadColumn(3)] public float DeltaT     { get; set; }
    [LoadColumn(4)] public bool  IsAnomaly  { get; set; }
}

public class AnomalyPrediction
{
    [ColumnName("PredictedLabel")]
    public bool  PredictedLabel { get; set; }
    public float Probability    { get; set; }
    public float Score          { get; set; }
}
