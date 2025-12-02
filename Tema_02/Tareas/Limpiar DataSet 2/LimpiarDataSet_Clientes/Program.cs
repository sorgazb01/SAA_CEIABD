using System.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

const string fileInputPath = "datos_clientes.csv";
const string fileOutputPath = "datos_clientes_preprocessed.csv";

MLContext mlContext = new();

TextLoader.Column[] columns = [
    new TextLoader.Column("ID_Cliente", DataKind.Single, 0),
    new TextLoader.Column("Edad", DataKind.Single, 1),
    new TextLoader.Column("Genero", DataKind.String, 2),
    new TextLoader.Column("Ingreso_Mensual_USD", DataKind.Single, 3),
    new TextLoader.Column("Producto_Preferido", DataKind.String, 4),
    new TextLoader.Column("Frecuencia_Compra_mensual", DataKind.Single, 5),
    new TextLoader.Column("Region", DataKind.String, 6)
];

var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options
{
    HasHeader = true,
    Separators = [','],
    TrimWhitespace = true,
    MissingRealsAsNaNs = true,
    Columns = columns
});

IDataView data = loader.Load(fileInputPath);

var pipeline = mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Edad", inputColumnName: "Edad", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
.Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Ingreso_Mensual_USD", inputColumnName: "Ingreso_Mensual_USD", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
.Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Frecuencia_Compra_mensual", inputColumnName: "Frecuencia_Compra_mensual", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))

.Append(mlContext.Transforms.Categorical.OneHotEncoding("Genero_Encoded", "Genero"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("Producto_Preferido_Encoded", "Producto_Preferido"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("Region_Encoded", "Region"))

.Append(mlContext.Transforms.NormalizeMinMax("Edad_MinMax", "Edad"))
.Append(mlContext.Transforms.NormalizeMinMax("Ingreso_Mensual_USD_MinMax", "Ingreso_Mensual_USD"))
.Append(mlContext.Transforms.NormalizeMinMax("Frecuencia_Compra_mensual_MinMax", "Frecuencia_Compra_mensual"))

.Append(mlContext.Transforms.NormalizeMeanVariance("Edad_ZScore", "Edad"))
.Append(mlContext.Transforms.NormalizeMeanVariance("Ingreso_Mensual_USD_ZScore", "Ingreso_Mensual_USD"))
.Append(mlContext.Transforms.NormalizeMeanVariance("Frecuencia_Compra_mensual_ZScore", "Frecuencia_Compra_mensual"))
.Append(mlContext.Transforms.Concatenate("Features", "Edad_ZScore", "Ingreso_Mensual_USD_ZScore", "Frecuencia_Compra_mensual_ZScore", "Genero_Encoded", "Producto_Preferido_Encoded","Region_Encoded"))
.Append(mlContext.Transforms.SelectColumns("ID_Cliente", "Features"));



ITransformer transformer = pipeline.Fit(data);
IDataView transformedData = transformer.Transform(data);
using (var fs = File.Open(fileOutputPath, FileMode.Create, FileAccess.Write))
{
    mlContext.Data.SaveAsText(transformedData, fs, separatorChar: ',', headerRow: true);
}