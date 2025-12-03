using System.Data;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

const string fileInputPath = "datos_vivienda_coste.csv";
const string fileOutputPath = "datos_vivienda_coste_preprocessed.csv";

MLContext mlContext = new();

TextLoader.Column[] columns = [
    new TextLoader.Column("ID_Vivienda", DataKind.Single, 0),
    new TextLoader.Column("Numero_de_Habitaciones", DataKind.Single, 1),
    new TextLoader.Column("Ubicacion", DataKind.String, 2),
    new TextLoader.Column("Tamaño_m2", DataKind.Single, 3),
    new TextLoader.Column("Tipo_de_Vivienda", DataKind.String, 4),
    new TextLoader.Column("Coste_USD", DataKind.Single, 5),
    new TextLoader.Column("Año_de_Construcción", DataKind.Single, 6)
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

var pipeline = mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Numero_de_Habitaciones", inputColumnName: "Numero_de_Habitaciones", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
.Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Tamaño_m2", inputColumnName: "Tamaño_m2", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
.Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Coste_USD", inputColumnName: "Coste_USD", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))   

.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Ubicacion_Encoded",inputColumnName: "Ubicacion"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Tipo_de_Vivienda_Encoded",inputColumnName: "Tipo_de_Vivienda"))

.Append(mlContext.Transforms.NormalizeMinMax("Numero_de_Habitaciones_MinMax", "Numero_de_Habitaciones"))
.Append(mlContext.Transforms.NormalizeMinMax("Tamaño_m2_MinMax", "Tamaño_m2"))
.Append(mlContext.Transforms.NormalizeMinMax("Coste_USD_MinMax", "Coste_USD"))

.Append(mlContext.Transforms.NormalizeMeanVariance("Numero_de_Habitaciones_ZScore", "Numero_de_Habitaciones"))
.Append(mlContext.Transforms.NormalizeMeanVariance("Tamaño_m2_ZScore", "Tamaño_m2"))
.Append(mlContext.Transforms.NormalizeMeanVariance("Coste_USD_ZScore", "Coste_USD"))

.Append(mlContext.Transforms.Concatenate("Features","Numero_de_Habitaciones_ZScore","Tamaño_m2_ZScore","Coste_USD_ZScore","Ubicacion_Encoded","Tipo_de_Vivienda_Encoded"))
.Append(mlContext.Transforms.SelectColumns("ID_Vivienda", "Features"));

ITransformer transformer = pipeline.Fit(data);
IDataView transformedData = transformer.Transform(data);

using (var fs = File.Open(fileOutputPath, FileMode.Create, FileAccess.Write))
{
    mlContext.Data.SaveAsText(transformedData, fs, separatorChar: ',', headerRow: true);
}
