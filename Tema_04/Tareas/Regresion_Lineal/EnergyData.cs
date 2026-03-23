using Microsoft.ML.Data;

namespace PCEI;

public class EnergyData
{
    [LoadColumn(0)] public float TempC { get; set; }
    [LoadColumn(1)] public float HumidityPct { get; set; }
    [LoadColumn(2)] public float PressureBar { get; set; }
    [LoadColumn(3)] public float LoadPct { get; set; }
    [LoadColumn(4)] public float Vibration { get; set; }
    [LoadColumn(5)] public float EnergyKWh { get; set; }
}

public class EnergyPrediction
{
    [ColumnName("Score")]
    public float Score { get; set; }
}