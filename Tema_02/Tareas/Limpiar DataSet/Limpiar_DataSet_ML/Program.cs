using System.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

const string fileInputPath = "data.csv";
const string fileOutputPath = "data_preprocessed_ML.csv";

MLContext mlContext = new();

TextLoader.Column[] columns = [
    new TextLoader.Column("ID", DataKind.Single, 0),
    new TextLoader.Column("Edad", DataKind.Single, 1),
    new TextLoader.Column("Género", DataKind.String, 2),
    new TextLoader.Column("Ingresos_Mensuales", DataKind.Single, 3),
    new TextLoader.Column("Gastos_Anuales", DataKind.Single, 4),
    new TextLoader.Column("Educación", DataKind.String, 5),
    new TextLoader.Column("Calificación_Crédito", DataKind.Single, 6),
    new TextLoader.Column("Tiempo_Empleo", DataKind.Single, 7)
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
.Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Ingresos_Mensuales", inputColumnName: "Ingresos_Mensuales", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
.Append(mlContext.Transforms.ReplaceMissingValues("Gastos_Anuales", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
.Append(mlContext.Transforms.ReplaceMissingValues("Calificación_Crédito", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
.Append(mlContext.Transforms.ReplaceMissingValues("Tiempo_Empleo", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))

.Append(mlContext.Transforms.Categorical.OneHotEncoding("Género_Encoded", "Género"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding("Educación_Encoded", "Educación"))


.Append(mlContext.Transforms.NormalizeMinMax("Edad_MinMax", "Edad"))
.Append(mlContext.Transforms.NormalizeMinMax("Ingresos_Mensuales_MinMax", "Ingresos_Mensuales"))
.Append(mlContext.Transforms.NormalizeMinMax("Gastos_Anuales_MinMax", "Gastos_Anuales"))
.Append(mlContext.Transforms.NormalizeMinMax("Calificación_Crédito_MinMax", "Calificación_Crédito"))
.Append(mlContext.Transforms.NormalizeMinMax("Tiempo_Empleo_MinMax", "Tiempo_Empleo"))

.Append(mlContext.Transforms.NormalizeMeanVariance("Edad_ZScore", "Edad"))
.Append(mlContext.Transforms.NormalizeMeanVariance("Ingresos_Mensuales_ZScore", "Ingresos_Mensuales"))
.Append(mlContext.Transforms.NormalizeMeanVariance("Gastos_Anuales_ZScore", "Gastos_Anuales"))
.Append(mlContext.Transforms.NormalizeMeanVariance("Calificación_Crédito_ZScore", "Calificación_Crédito"))
.Append(mlContext.Transforms.NormalizeMeanVariance("Tiempo_Empleo_ZScore", "Tiempo_Empleo"))

.Append(
    mlContext.Transforms.CustomMapping<CustomRow, CustomRowOut>(
        calcularRatioDeuda,
        contractName: null 
    )
)
);

ITransformer transformer = pipeline.Fit(data);
IDataView transformedData = transformer.Transform(data);
using (var fs = File.Open(fileOutputPath, FileMode.Create, FileAccess.Write))
{
    mlContext.Data.SaveAsText(transformedData, fs, separatorChar: ',', headerRow: true, schema: false);
}

void calcularRatioDeuda(CustomRow input, CustomRowOut output)
{
    float ingresosAnuales = input.Ingresos_Mensuales * 12;
    if (ingresosAnuales != 0)
        output.Ratio_Deuda = input.Gastos_Anuales / ingresosAnuales;
    else
        output.Ratio_Deuda = 0;
}

public class CustomRow
{
    public float Gastos_Anuales { get; set; }
    public float Ingresos_Mensuales { get; set; }
}
public class CustomRowOut
{
    public float Ratio_Deuda { get; set; }
}
