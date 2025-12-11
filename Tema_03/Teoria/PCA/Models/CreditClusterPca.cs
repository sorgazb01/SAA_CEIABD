using Microsoft.ML.Data;

namespace PCA.Models;

public class CreditClusterPca
{
    // 0,1,… rank-1
    [VectorType(2)]
    [ColumnName("PcaFeatures")]
    public float[]? PcaFeatures { get; set; }

    [ColumnName("PredictedLabel")]
    public uint PredictedClusterId { get; set; }

    public float[]? Score { get; set; }
}