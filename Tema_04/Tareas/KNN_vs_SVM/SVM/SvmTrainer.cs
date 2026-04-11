using Microsoft.ML.Data;

public class SvmDataPoint
{
    [VectorType(2)]
    public float[] Features { get; set; } = [];
    public bool Label { get; set; }
}

public class SvmPrediction
{
    public bool PredictedLabel { get; set; }
    public float Score { get; set; }
}
