using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

var mlContext = new MLContext(seed: 1);

var samples = new List<TextData>()
{
    // ==================================================
    // GAMEDEV / UNREAL (30)
    // ==================================================
    new() { Text = "Unreal Engine gameplay ability system and character abilities" },
    new() { Text = "Gameplay effects and abilities implemented using Unreal Engine GAS" },
    new() { Text = "Unreal Engine blueprint system for gameplay and combat abilities" },
    new() { Text = "Character gameplay logic using ability system in Unreal Engine" },
    new() { Text = "Combat gameplay abilities and effects in Unreal Engine projects" },
    new() { Text = "Unreal Engine gameplay framework for abilities and combat systems" },
    new() { Text = "Implementing gameplay abilities using Unreal Engine GAS framework" },
    new() { Text = "Gameplay systems and ability logic developed in Unreal Engine" },
    new() { Text = "Unreal Engine character abilities and gameplay effect pipelines" },
    new() { Text = "Combat systems using gameplay abilities in Unreal Engine" },
    new() { Text = "Unreal Engine gameplay programming with abilities and effects" },
    new() { Text = "Gameplay ability system architecture in Unreal Engine projects" },
    new() { Text = "Unreal Engine blueprint gameplay logic for ability based combat" },
    new() { Text = "Advanced gameplay ability systems implemented in Unreal Engine" },
    new() { Text = "Unreal Engine gameplay effects and ability execution flow" },

    new() { Text = "Configuring AbilitySystemComponent with attribute sets in Unreal Engine" },
    new() { Text = "Applying GameplayTags to gate abilities and effects in Unreal Engine GAS" },
    new() { Text = "Designing cooldowns costs and ability activation policies in GAS" },
    new() { Text = "Creating GameplayEffects for damage healing and buffs in Unreal Engine" },
    new() { Text = "Building ability tasks for targeting montages and root motion in GAS" },
    new() { Text = "Network replication for abilities and prediction in Unreal Engine GAS" },
    new() { Text = "Implementing ranged shooting ability with ammo consumption in GAS" },
    new() { Text = "Melee combo abilities with gameplay events and montage notifies in Unreal" },
    new() { Text = "Attribute replication and clamping health stamina and movement speed in GAS" },
    new() { Text = "Using Enhanced Input to trigger gameplay abilities in Unreal Engine" },
    new() { Text = "Creating target acquisition with line trace and hit results for abilities" },
    new() { Text = "Ability cancellation and blocking rules using gameplay tags in Unreal" },
    new() { Text = "Implementing dash dodge ability with cooldown and invulnerability frames" },
    new() { Text = "Building UI for cooldown timers and resource bars driven by attributes" },
    new() { Text = "Gameplay cue notifications for VFX SFX and camera shake in GAS abilities" },

    // ==================================================
    // DATABASE / SQL (30)
    // ==================================================
    new() { Text = "PostgreSQL functions procedures and exception handling" },
    new() { Text = "Database transactions rollback and exception handling in PostgreSQL" },
    new() { Text = "Stored procedures triggers and SQL functions in PostgreSQL" },
    new() { Text = "PostgreSQL schema design queries and database transactions" },
    new() { Text = "SQL queries functions and procedures with error handling" },
    new() { Text = "PostgreSQL transaction management and rollback strategies" },
    new() { Text = "SQL procedure execution and exception handling in PostgreSQL" },
    new() { Text = "Database schema constraints indexes and PostgreSQL queries" },
    new() { Text = "Advanced PostgreSQL functions triggers and procedures" },
    new() { Text = "PostgreSQL error handling within SQL transactions" },
    new() { Text = "SQL function performance tuning in PostgreSQL databases" },
    new() { Text = "PostgreSQL stored procedures and transactional control" },
    new() { Text = "Database design using PostgreSQL schemas and constraints" },
    new() { Text = "PostgreSQL query optimization and index strategies" },
    new() { Text = "SQL exception handling and transaction rollback mechanisms" },

    new() { Text = "PL/pgSQL exception blocks with RAISE NOTICE and RAISE EXCEPTION patterns" },
    new() { Text = "Using savepoints and partial rollback for robust PostgreSQL transactions" },
    new() { Text = "Creating roles users and grants for secure access control in PostgreSQL" },
    new() { Text = "Designing normalized tables with foreign keys and check constraints" },
    new() { Text = "Writing set returning functions with RETURNS TABLE in PostgreSQL" },
    new() { Text = "Implementing triggers for audit logging and change history tables" },
    new() { Text = "Optimizing queries with EXPLAIN ANALYZE and index selectivity" },
    new() { Text = "Creating views materialized views and using them for reporting in PostgreSQL" },
    new() { Text = "Handling unique constraint violations inside PL/pgSQL procedures" },
    new() { Text = "Using JSONB operators indexes and queries in PostgreSQL databases" },
    new() { Text = "Building enums domains and custom types for data integrity in PostgreSQL" },
    new() { Text = "Row level security policies for multi tenant databases in PostgreSQL" },
    new() { Text = "Using CTEs window functions and aggregation for analytical SQL queries" },
    new() { Text = "Maintaining database migrations and schema versions with SQL scripts" },
    new() { Text = "Writing idempotent DDL scripts with IF NOT EXISTS and safe ALTER TABLE" },

    // ==================================================
    // VERSION CONTROL / PERFORCE (30)
    // ==================================================
    new() { Text = "Version control using Perforce streams and asset locking" },
    new() { Text = "Perforce version control with streams workspaces and depots" },
    new() { Text = "Managing binary assets with Perforce locking and version control" },
    new() { Text = "Perforce workflow using streams branches and asset locking" },
    new() { Text = "Version control system using Perforce for large binary assets" },
    new() { Text = "Perforce streams workflow for game development assets" },
    new() { Text = "Asset locking and version control strategies using Perforce" },
    new() { Text = "Perforce depot management and workspace configuration" },
    new() { Text = "Version control pipelines using Perforce streams and branches" },
    new() { Text = "Binary asset management and locking with Perforce" },
    new() { Text = "Perforce stream based version control for large projects" },
    new() { Text = "Managing game assets with Perforce version control systems" },
    new() { Text = "Perforce workflows for asset locking and collaboration" },
    new() { Text = "Version control best practices using Perforce streams" },
    new() { Text = "Perforce version control for large binary asset pipelines" },

    new() { Text = "Creating Perforce streams and defining parent child stream relationships" },
    new() { Text = "Using p4ignore to exclude Intermediate Saved and DerivedDataCache folders" },
    new() { Text = "Resolving merges and integrating changes across Perforce streams safely" },
    new() { Text = "Workspace mapping and client views for large depots in Perforce" },
    new() { Text = "Handling file type settings like binary+l and text+k in Perforce" },
    new() { Text = "Submitting changelists with shelve review and promote workflows in Perforce" },
    new() { Text = "Using Perforce Swarm for code reviews and collaboration" },
    new() { Text = "Managing Unreal Engine projects with Perforce stream depots and locking" },
    new() { Text = "Branching strategy with main development and release streams in Perforce" },
    new() { Text = "Perforce server setup users groups protections and permissions model" },
    new() { Text = "Avoiding large binary merges by enforcing exclusive checkout in Perforce" },
    new() { Text = "Handling deleted files across streams and reconciling missing files in Perforce" },
    new() { Text = "Configuring Git LFS versus Perforce for large files and binary assets" },
    new() { Text = "Perforce reconcile offline work and detect moved renamed files in workspace" },
    new() { Text = "Managing changelist descriptions standards and tagging for traceability in Perforce" },

    // ==================================================
    // MACHINE LEARNING / LDA (30)
    // ==================================================
    new() { Text = "Topic modeling with LDA and probabilistic models" },
    new() { Text = "Latent Dirichlet Allocation for topic modeling in machine learning" },
    new() { Text = "Probabilistic topic modeling using LDA and text features" },
    new() { Text = "Machine learning topic modeling with LDA and LightLDA" },
    new() { Text = "Text analysis and topic modeling using LDA algorithms" },
    new() { Text = "Topic modeling techniques based on probabilistic latent variables" },
    new() { Text = "LDA topic modeling applied to text analytics tasks" },
    new() { Text = "Unsupervised topic modeling using Latent Dirichlet Allocation" },
    new() { Text = "Probabilistic generative models for topic discovery using LDA" },
    new() { Text = "Topic modeling pipelines using LDA in machine learning systems" },
    new() { Text = "Text feature extraction and topic modeling with LDA" },
    new() { Text = "Applying LDA for probabilistic topic extraction from documents" },
    new() { Text = "Machine learning workflows for topic modeling with LDA" },
    new() { Text = "Topic modeling with LightLDA in large text corpora" },
    new() { Text = "Advanced topic modeling using probabilistic LDA techniques" },

    new() { Text = "Tuning alpha and beta priors to control topic sparsity in LDA models" },
    new() { Text = "Comparing LDA topic coherence and interpretability across different topic counts" },
    new() { Text = "Using ngrams and stopwords to improve topic separation in LDA pipelines" },
    new() { Text = "Evaluating topic stability by retraining LDA with different random seeds" },
    new() { Text = "Extracting top terms per topic using LDA transformer model details" },
    new() { Text = "Understanding document topic distribution and topic word distribution in LDA" },
    new() { Text = "Using LightLDA sampling and iterations to improve convergence in ML.NET" },
    new() { Text = "Topic modeling for short texts and limitations of bag of words approaches" },
    new() { Text = "Preprocessing text with normalization tokenization and stopword removal for LDA" },
    new() { Text = "Using TF weighting rather than TF IDF for probabilistic topic modeling" },
    new() { Text = "Reducing vocabulary size to avoid sparse topics in LDA feature extraction" },
    new() { Text = "Diagnosing empty documents after stopword removal in text pipelines" },
    new() { Text = "Selecting the number of topics using held out likelihood and coherence measures" },
    new() { Text = "Topic modeling versus clustering embeddings for unsupervised text grouping" },
    new() { Text = "Building explainable topic summaries with top terms and topic probabilities" },
};



// Cargamos los datos (en este ejemplo desde memoria)
var dataview = mlContext.Data.LoadFromEnumerable(samples);

var options = new TextFeaturizingEstimator.Options
{
    CaseMode = TextNormalizingEstimator.CaseMode.Upper,
    KeepDiacritics = false,
    KeepNumbers = false,
    KeepPunctuations = false,
    Norm = TextFeaturizingEstimator.NormFunction.L1,
    CharFeatureExtractor = new WordBagEstimator.Options { },
    OutputTokensColumnName = "TokenizedText",
    StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options { },
    WordFeatureExtractor = new WordBagEstimator.Options { },
};

var a = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Tokens", options);

// PIPELINE para diapositivas
// var pipeline = mlContext.Transforms.Text.NormalizeText(outputColumnName: "NormalizedText", inputColumnName: "Text")
//     .Append(mlContext.Transforms.Text.TokenizeIntoWords(outputColumnName: "Tokens", inputColumnName: "NormalizedText"))
//     .Append(mlContext.Transforms.Text.RemoveDefaultStopWords(outputColumnName: "Tokens", inputColumnName: "Tokens"))
//     .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Tokens", inputColumnName: "Tokens"))
//     .Append(mlContext.Transforms.Text.ProduceNgrams(outputColumnName: "Tokens", inputColumnName: "Tokens")) // <----
//     .Append(mlContext.Transforms.Text.LatentDirichletAllocation(outputColumnName: "Features", inputColumnName: "Tokens", numberOfTopics: 4));

// Stopwords de DOMINIO (son aquellas palabras que ignoramos)
var domainStop = new[]
{
    "using",
    "system",
    "topic",
    "model",
    "models",
    "handling",
    "version",
    "control",
    // etc.
};

var pipeline =
    // Limpiamos el texto
    mlContext.Transforms.Text.NormalizeText(
        outputColumnName: "NormText",
        inputColumnName: "Text",
        caseMode: TextNormalizingEstimator.CaseMode.Lower,
        keepDiacritics: false,
        keepNumbers: true,
        keepPunctuations: false)

    // Tokenizamos
    .Append(mlContext.Transforms.Text.TokenizeIntoWords(
        outputColumnName: "Words",
        inputColumnName: "NormText"))

    // Stopwords del idioma + las de dominio
    .Append(mlContext.Transforms.Text.RemoveDefaultStopWords(
        outputColumnName: "WordsNoStop",
        inputColumnName: "Words",
        language: StopWordsRemovingEstimator.Language.English)) // <--- o Spanish si analizamos nuestro idioma
    .Append(mlContext.Transforms.Text.RemoveStopWords(
        outputColumnName: "WordsNoStop",
        inputColumnName: "WordsNoStop",
        stopwords: domainStop)) // <--- incluimos las nuestras de dominio

    // Convertir tokens a KeyType (necesario para n-grams "por vocabulario")
    .Append(mlContext.Transforms.Conversion.MapValueToKey(
        outputColumnName: "WordKeys",
        inputColumnName: "WordsNoStop"))

    // N-grams
    .Append(mlContext.Transforms.Text.ProduceNgrams(
        outputColumnName: "Ngrams",
        inputColumnName: "WordKeys",
        ngramLength: 1, // <--- podemos jugar con este valor
        useAllLengths: false, // <--- incluir todos o solo lo definidos en ngramLength
        weighting: NgramExtractingEstimator.WeightingCriteria.Tf)) // <--- Define como se pondera cada n-grama; para topic modeling es mejor Tf

    // Unigram (n=1): ["unreal"], ["postgresql"]
    // Bigram (n=2): ["gameplay ability"], ["exception handling"]
    // Trigram (n=3): ["gameplay ability system"]

    // LDA: entrada vector<float> (Ngrams) -> salida vector<float> (Features)
    .Append(mlContext.Transforms.Text.LatentDirichletAllocation(
        outputColumnName: "Features",
        inputColumnName: "Ngrams",
        numberOfTopics: 4, // <--- Cuántos topicos queremos? Número bajo buscamos tópicos generales. E inversa.
        alphaSum: 0.01f, // mezcla de tópicos por documento. Bajo = pocos topicos por doc.
        beta: 0.01f, // mezcla de palabras por tópico. Bajo = un tópico usa pocas palabras.
        samplingStepCount: 4, // si vemos resultados raros, vamos subiendo
        maximumNumberOfIterations: 5000, // vamos subiendo hasta que vemos que no converge
        maximumTokenCountPerDocument: 5, // cuantos tokens representativos por doc
        numberOfSummaryTermsPerTopic: 1 // palabras por topico
        ));

// Entrenamos el modelo
var transformer = pipeline.Fit(dataview);

// Motor de predicciones
var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(transformer);

Console.WriteLine("==============================");
Console.WriteLine("Topic1  Topic2  Topic3  Topic4");
PrintLdaFeatures(predictionEngine.Predict(samples[5]));
PrintLdaFeatures(predictionEngine.Predict(samples[35]));
PrintLdaFeatures(predictionEngine.Predict(samples[65]));
PrintLdaFeatures(predictionEngine.Predict(samples[95]));

Console.WriteLine();
Console.WriteLine("==============================");
Console.WriteLine("Topic1  Topic2  Topic3  Topic4");
PrintLdaFeatures(predictionEngine.Predict(new() { Text = "Unreal Engine for combat systems." }));
PrintLdaFeatures(predictionEngine.Predict(new() { Text = "PostgreSQL have stored functions in SQL." }));
PrintLdaFeatures(predictionEngine.Predict(new() { Text = "Perforce is a sersion control, with streams and workspaces." }));
PrintLdaFeatures(predictionEngine.Predict(new() { Text = "LDA is a probabilistic model." }));

// SALIDA PARA 4 TOPICOS:

// Topic1  Topic2  Topic3  Topic4
// 0.7500  0.2500  0.0000  0.0000  Unreal Engine for combat systems.
// 0.7500  0.0000  0.0000  0.2500  PostgreSQL have stored functions in SQL.
// 0.6667  0.0000  0.3333  0.0000  Perforce is a sersion control, with streams and workspaces.
// 0.0000  0.0000  1.0000  0.0000  LDA is a probabilistic model.

// Aquí viene lo complicado ... mi interpretación del segundo grupo:
// | Tema   | Interpretación humana         |
// | ------ | ----------------------------- |
// |   T1   | Motor                         |
// |   T2   | Gameplay                      |
// |   T3   | Modelos / ficheros            |
// |   T4   | SQL                           |


static void PrintLdaFeatures(TransformedTextData prediction)
{
    for (int i = 0; i < prediction.Features.Length; i++)
    {
        Console.Write($"{prediction.Features[i]:F4}  ");
    }
    Console.WriteLine($"{prediction.Text}  ");
}

class TextData
{
    public string Text { get; set; }
}

class TransformedTextData : TextData
{
    public float[] Features { get; set; }
}