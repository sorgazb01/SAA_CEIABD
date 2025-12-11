using Microsoft.ML;
using Microsoft.ML.Transforms;
using PCA.Models;

//////////////////
////// Contexto
//////////////////

var mlContext = new MLContext(seed: 1);

const string DatasetPath = "Data/credit.csv";


//////////////////
////// Cargamos los datos
//////////////////

IDataView dataView = mlContext.Data.LoadFromTextFile<CreditData>(
    path: DatasetPath,
    hasHeader: true,
    separatorChar: ',');

ShowData<CreditData>(mlContext, dataView, 5);


//////////////////
////// Pipeline de PROCESAMIENTO
//////////////////

var preprocessingPipeline =
    mlContext.Transforms.Categorical.OneHotEncoding(
        [
            new InputOutputColumnPair("GeneroEncoded", nameof(CreditData.Genero)),
            new InputOutputColumnPair("EducacionEncoded", nameof(CreditData.Educacion))
        ],
        outputKind: OneHotEncodingEstimator.OutputKind.Indicator)
    .Append(mlContext.Transforms.Concatenate("Features",
        [nameof(CreditData.Edad),
        nameof(CreditData.Ingresos_Mensuales),
        nameof(CreditData.Gastos_Anuales),
        nameof(CreditData.Calificacion_Credito),
        nameof(CreditData.Tiempo_Empleo),
        nameof(CreditData.Ratio_Deuda),
        "GeneroEncoded",
        "EducacionEncoded"]))
    // Opcional pero recomendable antes de PCA
    .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"));


//////////////////
////// Pipeline de REDUCCIÓN DE LA DIMENSIÓN
//////////////////

var pcaPipeline =
    preprocessingPipeline.Append(
        mlContext.Transforms.ProjectToPrincipalComponents(
            outputColumnName: "PcaFeatures",
            inputColumnName: "Features",
            rank: 2
        ));


//////////////////
////// Pipeline de CLUSTERING
//////////////////

var clusteringPipeline =
    pcaPipeline.Append(
        mlContext.Clustering.Trainers.KMeans(
            featureColumnName: "PcaFeatures",
            numberOfClusters: 4
        ));


//////////////////
////// Creación del modelo
////// M = {PREPRO + PCA + KMEAN}
//////////////////

var model = clusteringPipeline.Fit(dataView);

// TODO: consumir el modelo


//////////////////
////// Obtener datos transformados (PCA)
////// Mpca = {PREPRO + PCA}
//////////////////

var pcaModel = pcaPipeline.Fit(dataView);


//////////////////
////// Transformar datos
//////////////////

var pcaTransformed = pcaModel.Transform(dataView);

ShowData<PcaOnly>(mlContext, pcaTransformed, 5);


static void ShowData<TRow>(MLContext mlContext, IDataView dataView, int count)
    where TRow : class, new()
{
    var preview = mlContext.Data.CreateEnumerable<TRow>(dataView, reuseRowObject: false)
                                .Take(count)
                                .ToList();

    foreach (var row in preview)
    {
        Console.WriteLine(row);
    }
}