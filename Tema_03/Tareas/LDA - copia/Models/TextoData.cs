using Microsoft.ML.Data;
namespace LDA.Models;

class TextoData
{
    public string? Texto { get; set;}
}

class TextoDataTransformado : TextoData
{
    public float[] Features { get; set;} = Array.Empty<float>();
}