using Microsoft.ML.Data;

namespace ComercialOffer;

public class CustomerData
{
    [LoadColumn(0)]
    public float Age { get; set; }
    [LoadColumn(1)]
    public float Income { get; set; }
    [LoadColumn(2)]
    public float PreviousPurchases { get; set; }
    [LoadColumn(3)]
    public float WebVisits { get; set; }
    [LoadColumn(4)]
    public bool SubcribedNewsletter { get; set; }
    [LoadColumn(5)]
    public string Region { get; set; } = string.Empty;
    [LoadColumn(6)]
    public bool Label { get; set;}
}

public class CustomerPrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }
    public float Score { get; set; }
}
