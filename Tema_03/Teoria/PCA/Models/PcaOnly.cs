using System.Globalization;
using Microsoft.ML.Data;

namespace PCA.Models;

/// <summary>
/// Lo usamos para obtener las features reducidas.
/// </summary>
public class PcaOnly
{
    [VectorType(2)]
    public float[]? PcaFeatures { get; set; }

    public override string ToString()
    {
        string pca = PcaFeatures is null
            ? "null"
            : string.Join(", ", PcaFeatures.Select(v => v.ToString("0.###", CultureInfo.InvariantCulture)));

        return pca;
    }
}