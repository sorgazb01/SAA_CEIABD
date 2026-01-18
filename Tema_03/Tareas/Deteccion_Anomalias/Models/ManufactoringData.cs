using Microsoft.ML.Data;
namespace Deteccion_Anomalias.Models;

public class ManufactoringData
{
    [LoadColumn(0)]
    public float Id { get; set; } 

    [LoadColumn(1)]
    public float Weight { get; set; }

    [LoadColumn(2)]
    public float Length { get; set; }

    [LoadColumn(3)]
    public float Width { get; set; }

    [LoadColumn(4)]
    public float Hardness { get; set; }

    [LoadColumn(5)]
    public float TempProcess { get; set; }

    [LoadColumn(6)]
    public float VibrationRms { get; set; }

    [LoadColumn(7)]
    public float Label { get; set; }
}

public class ManufactoringPrediction : ManufactoringData
{
    public bool PredictedLabel { get; set; }

    public float Score { get; set; }
}