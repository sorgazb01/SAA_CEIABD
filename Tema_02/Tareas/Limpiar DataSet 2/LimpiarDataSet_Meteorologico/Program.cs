using System.Data;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

const string fileInputPath = "datos_meteorologicos.csv";
const string fileOutputPath = "datos_meteorologicos_preprocessed.csv";

MLContext mlContext = new();

TextLoader.Column[] columns = [
    new TextLoader.Column("Fecha", DataKind.DateTime, 0),
    new TextLoader.Column("Temperatura_C", DataKind.Double, 1),
    new TextLoader.Column("Humedad_%", DataKind.Double, 2),
    new TextLoader.Column("Tipo_de_Clima", DataKind.String, 3),
    new TextLoader.Column("Velocidad_Viento_kmh", DataKind.Double, 4),
    new TextLoader.Column("Precipitacion_mm", DataKind.Double, 5),
    new TextLoader.Column("Presion_hPa", DataKind.Double, 6)
];

var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options
{
    HasHeader = true,
    Separators = new[] { ',' },
    TrimWhitespace = true,
    MissingRealsAsNaNs = true,
    Columns = columns
});

IDataView data = loader.Load(fileInputPath);

var pipeline = mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Temperatura_C", inputColumnName: "Temperatura_C",replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
.Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Humedad_%", inputColumnName: "Humedad_%", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
.Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Velocidad_Viento_kmh", inputColumnName: "Velocidad_Viento_kmh", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
.Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Precipitacion_mm", inputColumnName: "Precipitacion_mm", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
.Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Presion_hPa", inputColumnName: "Presion_hPa", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))

.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Tipo_de_Clima_Encoded",inputColumnName: "Tipo_de_Clima"))

.Append(mlContext.Transforms.Conversion.ConvertType(outputColumnName: "Tipo_de_Clima_Encoded_Double", inputColumnName: "Tipo_de_Clima_Encoded", outputKind: DataKind.Double))

.Append(mlContext.Transforms.NormalizeMinMax("Temperatura_C_MinMax", "Temperatura_C"))
.Append(mlContext.Transforms.NormalizeMinMax("Humedad_%_MinMax", "Humedad_%"))
.Append(mlContext.Transforms.NormalizeMinMax("Velocidad_Viento_kmh_MinMax", "Velocidad_Viento_kmh"))
.Append(mlContext.Transforms.NormalizeMinMax("Precipitacion_mm_MinMax", "Precipitacion_mm"))
.Append(mlContext.Transforms.NormalizeMinMax("Presion_hPa_MinMax", "Presion_hPa"))

    
.Append(mlContext.Transforms.NormalizeMeanVariance("Temperatura_C_ZScore", "Temperatura_C"))
.Append(mlContext.Transforms.NormalizeMeanVariance("Humedad_%_ZScore", "Humedad_%"))
.Append(mlContext.Transforms.NormalizeMeanVariance("Velocidad_Viento_kmh_ZScore", "Velocidad_Viento_kmh"))
.Append(mlContext.Transforms.NormalizeMeanVariance("Precipitacion_mm_ZScore", "Precipitacion_mm"))
.Append(mlContext.Transforms.NormalizeMeanVariance("Presion_hPa_ZScore", "Presion_hPa"))

.Append(mlContext.Transforms.Concatenate("Features", "Temperatura_C_ZScore", "Humedad_%_ZScore", "Velocidad_Viento_kmh_ZScore", "Precipitacion_mm_ZScore", "Presion_hPa_ZScore", "Tipo_de_Clima_Encoded_Double"))

.Append(mlContext.Transforms.SelectColumns("Fecha", "Features"));

ITransformer transformer = pipeline.Fit(data);
IDataView transformedData = transformer.Transform(data);
using (var fs = File.Open(fileOutputPath, FileMode.Create, FileAccess.Write))
{
    mlContext.Data.SaveAsText(transformedData, fs, separatorChar: ',', headerRow: true);
}
