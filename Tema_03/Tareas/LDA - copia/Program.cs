using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using LDA.Models;

var mlContext = new MLContext(seed: 1);

// Opcion de Lista
// ----------------------------------------------------------------------------------------------------------------------------------------------
// var ejemplos = new List<TextoData>()
// {
//     new() { Texto = "El paquete llegó dos días tarde y la caja venía golpeada. Los auriculares funcionan pero me quedé con mala impresión." },
//     new() { Texto = "Zapatillas cómodas y buena calidad pero el envío fue lentísimo." },
//     new() { Texto = "La cafetera tiene buenos materiales y calienta rápido. El precio me parece justo para lo que ofrece." },
//     new() { Texto = "La mochila llegó con la cremallera atascada y atención al cliente tardó en responder." },
//     new() { Texto = "Smartphone con buena pantalla pero vino sin el precinto original. La entrega fue puntual eso sí." },
//     new() { Texto = "Silla de oficina fácil de montar y muy estable. Por el precio esperaba algo peor." },
//     new() { Texto = "El repartidor dejó el paquete en otro portal y tuve que buscarlo. Producto correcto." },
//     new() { Texto = "Auriculares con sonido decente pero demasiados plásticos para lo que cuestan." },
//     new() { Texto = "Pedí devolución y el reembolso tardó más de una semana. Atención al cliente correcta pero lenta." },
//     new() { Texto = "Zapatillas bonitas pero talla pequeña. Devolver fue sencillo y me devolvieron el dinero bien." },
//     new() { Texto = "La cafetera gotea por la base. El servicio de soporte me pidió vídeos y aún no solucionan nada." },
//     new() { Texto = "La mochila es espaciosa y las costuras se ven firmes. Llegó al día siguiente." },
//     new() { Texto = "Smartphone rápido y sin fallos. Precio algo alto pero la calidad acompaña." },
//     new() { Texto = "La silla hace ruido al reclinarse y vino con un tornillo mal. Envío rápido pero embalaje pobre." },
//     new() { Texto = "Auriculares excelentes para llamadas. Atención al cliente me ayudó a cambiar la dirección antes del envío." },
//     new() { Texto = "Zapatillas con suela cómoda pero olor fuerte al abrir. Entrega correcta." },
//     new() { Texto = "La cafetera llegó rota y el embalaje era insuficiente. Estoy esperando la recogida para la devolución." },
//     new() { Texto = "Mochila ligera y resistente al agua. Muy buena relación calidad precio." },
//     new() { Texto = "Smartphone con batería floja para el precio. La entrega fue puntual." },
//     new() { Texto = "La silla es cómoda pero el asiento es más duro de lo esperado. Buen precio en oferta." },
//     new() { Texto = "Envío en fecha pero caja abierta. Auriculares funcionan aunque no me fío." },
//     new() { Texto = "Zapatillas con acabados pobres y costuras torcidas. Devolución sin problemas." },
//     new() { Texto = "Cafetera compacta y bonita. Atención al cliente respondió rápido a una duda." },
//     new() { Texto = "Mochila con bolsillos bien pensados. Por el precio está genial." },
//     new() { Texto = "Smartphone llegó tarde y sin aviso. Atención al cliente solo dijo que esperara." },
//     new() { Texto = "Silla de oficina robusta. Montaje sencillo y materiales correctos." },
//     new() { Texto = "Auriculares con buen bajo. Pero por ese precio esperaba mejor cancelación." },
//     new() { Texto = "Zapatillas llegaron con manchas. Me ofrecieron cambio y llegó el reemplazo rápido." },
//     new() { Texto = "La cafetera hace mucho ruido y tarda en calentar. El precio no lo justifica." },
//     new() { Texto = "Mochila con cremalleras suaves. Envío rápido y embalaje perfecto." },
//     new() { Texto = "Smartphone excelente calidad pero la entrega fue un desastre. Llegó una semana tarde." },
//     new() { Texto = "La silla vino con una pieza rota y gestionar garantía fue lento." },
//     new() { Texto = "Auriculares cómodos para horas. Entrega al día siguiente." },
//     new() { Texto = "Zapatillas con buen agarre. El precio es bueno y la calidad también." },
//     new() { Texto = "Cafetera con fugas desde el primer día. Atención al cliente me mareó con formularios." },
//     new() { Texto = "La mochila parece resistente pero el envío llegó aplastado. Por suerte no se dañó." },
//     new() { Texto = "Smartphone con cámara muy buena. Relación calidad precio sobresaliente." },
//     new() { Texto = "Silla de oficina con ruedas malas. Devolver fue fácil y rápido." },
//     new() { Texto = "Auriculares llegaron tarde y encima venían sin almohadillas extra. Muy mal." },
//     new() { Texto = "Zapatillas cómodas pero el color no es como en la foto. Atención al cliente aceptó devolución sin pegas." },
//     new() { Texto = "Cafetera pequeña ideal para una persona. Envío rápido." },
//     new() { Texto = "Mochila grande y cómoda. Precio alto pero se nota la calidad." },
//     new() { Texto = "Smartphone se calienta bastante. Servicio al cliente me respondió en 24 horas." },
//     new() { Texto = "Silla firme y buena para espalda. Por lo que cuesta está bien." },
//     new() { Texto = "El paquete llegó con el precinto roto. Auriculares bien pero no repetiría por el envío." },
//     new() { Texto = "Zapatillas con buena amortiguación. Envío en fecha y sin daños." },
//     new() { Texto = "Cafetera con plástico endeble. Precio demasiado alto." },
//     new() { Texto = "Mochila con tirantes cómodos. Devolución rápida cuando me equivoqué de modelo." },
//     new() { Texto = "Smartphone tardó en llegar y nadie daba información. Atención al cliente pésima." },
//     new() { Texto = "Silla llegó con arañazos. Me ofrecieron reembolso parcial y acepté." },
//     new() { Texto = "Auriculares con micrófono muy flojo. Devolver fue sencillo." },
//     new() { Texto = "Zapatillas perfectas para correr. Calidad buena y precio razonable." },
//     new() { Texto = "Cafetera llegó con la caja húmeda. Funciona pero me preocupa." },
//     new() { Texto = "Mochila bonita pero la cremallera principal se rompió. Garantía lenta." },
//     new() { Texto = "Smartphone excelente por el precio. Entrega rápida." },
//     new() { Texto = "Silla muy cómoda pero instrucciones confusas. Atención al cliente me envió un vídeo útil." },
//     new() { Texto = "Auriculares con gran cancelación. Llegaron tarde pero valen la pena." },
//     new() { Texto = "Zapatillas con suela dura. Precio barato y se nota." },
//     new() { Texto = "Cafetera hace café aguado. El soporte me pidió limpieza y sigue igual." },
//     new() { Texto = "Mochila resistente y ligera. Envío rápido." },
//     new() { Texto = "Smartphone vino con cargador incorrecto. Atención al cliente lo solucionó en dos días." },
//     new() { Texto = "La silla cruje al moverse. Devolución aceptada sin complicaciones." },
//     new() { Texto = "Auriculares con sonido plano. Por el precio esperaba más." },
//     new() { Texto = "Zapatillas cómodas aunque el acabado interior roza. Llegaron en fecha." },
//     new() { Texto = "Cafetera excelente y fácil de limpiar. Relación calidad precio muy buena." },
//     new() { Texto = "Mochila llegó tarde y el paquete venía roto. Dentro estaba todo bien." },
//     new() { Texto = "Smartphone con buena pantalla pero batería normalita. Precio justo." },
//     new() { Texto = "Silla estable y buena para teletrabajo. Envío rapidísimo." },
//     new() { Texto = "Auriculares llegaron sin manual. Atención al cliente me lo mandó por email." },
//     new() { Texto = "Zapatillas se despegaron en una semana. Estoy tramitando devolución." },
//     new() { Texto = "Cafetera con depósito pequeño. Por el precio está aceptable." },
//     new() { Texto = "Mochila con costuras débiles. Devolver fue fácil." },
//     new() { Texto = "Smartphone llegó con golpe en una esquina. Embalaje fatal y soporte lento." },
//     new() { Texto = "Silla cómoda pero respaldo bajo. Precio bien en oferta." },
//     new() { Texto = "Auriculares cómodos pero se escucha ruido de fondo. Servicio al cliente no ayuda." },
//     new() { Texto = "Zapatillas bonitas y de buena calidad. Entrega perfecta." },
//     new() { Texto = "Cafetera tarda mucho en calentar. Atención al cliente correcta pero lenta." },
//     new() { Texto = "Mochila es enorme y cabe todo. Calidad buena." },
//     new() { Texto = "Smartphone por el precio está genial. Envío tardó un poco." },
//     new() { Texto = "La silla vino sin tornillos. Atención al cliente me envió repuesto rápido." },
//     new() { Texto = "Auriculares con buen sonido. Precio alto pero buen producto." },
//     new() { Texto = "Zapatillas llegaron con caja destrozada. El producto estaba bien." },
//     new() { Texto = "Cafetera dejó de funcionar al tercer uso. Devolución y reembolso ok." },
//     new() { Texto = "Mochila con olor fuerte a plástico. Envío rápido." },
//     new() { Texto = "Smartphone con cámara mediocre. Calidad precio no me convence." },
//     new() { Texto = "Silla de oficina buena pero tela calurosa. Entrega en fecha." },
//     new() { Texto = "Auriculares fallan al conectar por Bluetooth. Soporte me pidió reiniciar y nada." },
//     new() { Texto = "Zapatillas cómodas pero se ensucian fácil. Buen precio." },
//     new() { Texto = "Cafetera excelente espuma. Envío rápido y embalaje perfecto." },
//     new() { Texto = "Mochila con cremalleras flojas. Atención al cliente me ofreció cambio." },
//     new() { Texto = "Smartphone con buen rendimiento. Precio correcto." },
//     new() { Texto = "Silla estable pero reposabrazos endebles. Por el precio está bien." },
//     new() { Texto = "Auriculares llegaron tarde y sin seguimiento. Muy mala experiencia." },
//     new() { Texto = "Zapatillas con talla correcta. Calidad buena y entrega rápida." },
//     new() { Texto = "Cafetera vino con piezas sueltas. Atención al cliente respondió y mandó recambio." },
//     new() { Texto = "Mochila con buen acolchado. Precio alto pero cómoda." },
//     new() { Texto = "Smartphone se reinicia solo. Trámite de garantía lento." },
//     new() { Texto = "Silla llegó con la base rayada. Me devolvieron parte del dinero." },
//     new() { Texto = "Auriculares con sonido increíble. Lástima que el envío se retrasó." },
//     new() { Texto = "Zapatillas se ven baratas. Por el precio tampoco esperaba más." },
//     new() { Texto = "Cafetera compacta y buena. Precio justo." },
//     new() { Texto = "Mochila llegó tarde y la etiqueta estaba mal. Atención al cliente lo arregló." },
//     new() { Texto = "Smartphone muy bueno por el precio. Entrega en 24 horas." },
//     new() { Texto = "Silla cómoda pero tornillos de mala calidad. Envío correcto." },
//     new() { Texto = "Auriculares con cable corto. Atención al cliente no ofreció solución." },
//     new() { Texto = "Zapatillas con buen material y cosido. Buen valor por el precio." },
//     new() { Texto = "Cafetera gotea al servir. Solicité devolución y fue rápido." },
//     new() { Texto = "Mochila bonita pero poca capacidad. Precio algo alto." },
//     new() { Texto = "Smartphone vino con idioma mal configurado. Soporte me guió y bien." },
//     new() { Texto = "Silla de oficina excelente para espalda. Calidad alta." },
//     new() { Texto = "Auriculares con volumen bajo. Por el precio es decepcionante." },
//     new() { Texto = "Zapatillas llegaron usadas. Atención al cliente pidió fotos y me cambiaron el par." },
//     new() { Texto = "Cafetera buena pero manual malo. Envío rápido." },
//     new() { Texto = "Mochila con costuras perfectas. Muy buena relación calidad precio." },
//     new() { Texto = "Smartphone llegó tarde y sin protector. Producto bien." },
//     new() { Texto = "Silla cruje pero es cómoda. Precio razonable." },
//     new() { Texto = "Auriculares llegaron dañados y el reembolso tardó. Atención al cliente lenta." },
//     new() { Texto = "Zapatillas cómodas aunque el tejido es fino. Envío en fecha." },
//     new() { Texto = "Cafetera de buena calidad y fácil de usar. Precio alto pero vale." },
//     new() { Texto = "Mochila llegó con un tirante descosido. Devolución sin problema." },
//     new() { Texto = "Smartphone con buena cámara. Atención al cliente respondió rápido a dudas." },
//     new() { Texto = "Silla estable y ruedas suaves. Entrega muy rápida." },
//     new() { Texto = "Auriculares bien pero el estuche se raya fácil. Por el precio esperaba más." },
//     new() { Texto = "Zapatillas perfectas y cómodas. Muy buena compra." },
//     new() { Texto = "Cafetera no encaja bien el depósito. Soporte tardó en contestar." },
//     new() { Texto = "Mochila amplia y resistente. Llegó en perfecto estado." },
//     new() { Texto = "Smartphone se queda sin batería rápido. Para el precio es flojo." },
//     new() { Texto = "Silla buena pero el respaldo se mueve. Devolver fue sencillo." },
//     new() { Texto = "Auriculares llegaron con retraso y caja aplastada. Funcionan bien." },
//     new() { Texto = "Zapatillas con suela resbaladiza. Calidad mala." },
//     new() { Texto = "Cafetera excelente. Envío rápido y atención al cliente correcta." },
//     new() { Texto = "Mochila con bolsillos útiles. Precio muy bueno." },
//     new() { Texto = "Smartphone llegó con pantalla con píxel muerto. Cambio rápido por garantía." },
//     new() { Texto = "Silla cómoda pero reposabrazos bajos. Precio aceptable." },
//     new() { Texto = "Auriculares con gran sonido y buen micrófono. Entrega puntual." },
//     new() { Texto = "Zapatillas bonitas pero caras para lo que son. Devolución fácil." },
//     new() { Texto = "Cafetera hace café bien pero pierde agua. Estoy gestionando garantía." },
//     new() { Texto = "Mochila llegó tarde. Producto bien pero logística mal." },
//     new() { Texto = "Smartphone muy rápido. Relación calidad precio excelente." },
//     new() { Texto = "Silla de oficina con materiales pobres. Por el precio es una decepción." },
//     new() { Texto = "Auriculares con conexión estable. Precio correcto." },
//     new() { Texto = "Zapatillas llegaron con cordones diferentes. Atención al cliente lo solucionó." },
//     new() { Texto = "Cafetera compacta. Envío rápido y bien embalado." },
//     new() { Texto = "Mochila es ligera pero se mancha fácil. Precio ok." },
//     new() { Texto = "Smartphone con buen sonido. Pero el envío tardó y no avisaron." },
//     new() { Texto = "Silla llegó con piezas faltantes. Atención al cliente mandó repuestos." },
//     new() { Texto = "Auriculares con cancelación floja. Caros para lo que ofrecen." },
//     new() { Texto = "Zapatillas perfectas para caminar. Buena calidad." },
//     new() { Texto = "Cafetera dejó de calentar. Devolución rápida y reembolso en pocos días." },
//     new() { Texto = "Mochila robusta y cómoda. Precio alto pero se nota." },
//     new() { Texto = "Smartphone con cámara decente. Atención al cliente bien." },
//     new() { Texto = "Silla cómoda y buena para estudiar. Envío en fecha." },
//     new() { Texto = "Auriculares llegaron tarde y no contestan al soporte. Muy mal." },
//     new() { Texto = "Zapatillas se ajustan bien. Precio justo." },
//     new() { Texto = "Cafetera llegó con la caja rota. Producto funciona pero quedó feo." },
//     new() { Texto = "Mochila con cremalleras resistentes. Muy buena compra." },
//     new() { Texto = "Smartphone vino sin factura. Atención al cliente la envió al momento." },
//     new() { Texto = "Silla cruje en la base. Precio barato y se nota." },
//     new() { Texto = "Auriculares cómodos y buen sonido. Envío rápido." },
//     new() { Texto = "Zapatillas con mala calidad y pegamento visible. Devolver fue fácil." },
//     new() { Texto = "Cafetera buena pero piezas delicadas. Precio correcto." },
//     new() { Texto = "Mochila llegó con retraso y sin aviso. Producto correcto." },
//     new() { Texto = "Smartphone excelente pero caro. La calidad es alta." },
//     new() { Texto = "Silla estable y buen respaldo. Muy recomendable." },
//     new() { Texto = "Auriculares fallan al cargar. Trámite de garantía lento." },
//     new() { Texto = "Zapatillas cómodas y bonitas. Entrega perfecta." },
//     new() { Texto = "Cafetera con buena presión. Envío rápido." },
//     new() { Texto = "Mochila de buena calidad. Atención al cliente respondió rápido." },
//     new() { Texto = "Smartphone llegó tarde y con caja golpeada. Cambio correcto al final." },
//     new() { Texto = "Silla de oficina cómoda. Pero el embalaje venía abierto." },
//     new() { Texto = "Auriculares con buen sonido pero mala construcción. Por el precio no compensa." },
//     new() { Texto = "Zapatillas llegaron rápido y son cómodas. Buena relación calidad precio." },
//     new() { Texto = "Cafetera gotea y el soporte no soluciona. Pedí devolución." },
//     new() { Texto = "Mochila espaciosa. Precio buenísimo." },
//     new() { Texto = "Smartphone con pantalla excelente. Envío rápido." },
//     new() { Texto = "Silla llegó con defecto y el reembolso tardó. Atención al cliente lenta." },
//     new() { Texto = "Auriculares perfectos. Llegaron al día siguiente." },
//     new() { Texto = "Zapatillas con suela que se desgasta rápido. Calidad mala." },
//     new() { Texto = "Cafetera buena pero cara. Aun así cumple." },
//     new() { Texto = "Mochila llegó dañada y la devolución fue fácil. El reembolso tardó unos días." },
// };
// ----------------------------------------------------------------------------------------------------------------------------------------------

// Opcion ficheros txt
// -------------------------------------------
var resenias_01 = "Data/resenias_01.txt";
var resenias_02 = "Data/resenias_02.txt";

var ejemplos = File.ReadLines(resenias_01)
    .Concat(File.ReadLines(resenias_02))
    .Select(t => t.Trim())
    .Where(t => !string.IsNullOrWhiteSpace(t))
    .Select(t => new TextoData { Texto = t })
    .ToList();
// -------------------------------------------

// Cargo los datos
var dataview = mlContext.Data.LoadFromEnumerable(ejemplos);

// Opciones domainStopWords para la version de datos en Lista
// -------------------------------------------
// var domainStop = new[]
// {
//     "producto","paquete","pedido","compré",
//     "comprar","llegó","recibí","tienda",
//     "web"
// };
// -------------------------------------------

// Defino las StopWords
// -------------------------------------------
var domainStop = new[]
{
  "producto","productos","articulo","articulos","pedido",
  "reembolso","garantia","atencion","cliente","soporte",
  "me","mi","mis","yo","conmigo"
};
// -------------------------------------------


// PipeLine
var pipeline = 
    // Normalizo el texto con minusculas, sin tildes, sin signos de puntuacion
    mlContext.Transforms.Text.NormalizeText(
        outputColumnName: "NormText",
        inputColumnName: "Texto",
        caseMode: TextNormalizingEstimator.CaseMode.Lower,
        keepDiacritics: false,
        keepNumbers: true,
        keepPunctuations: false
    )
    // Tokenizacion
    .Append(
        mlContext.Transforms.Text.TokenizeIntoWords(
            outputColumnName: "Words",
            inputColumnName: "NormText"
        )
    )
    // StopWords por defecto del español
    .Append(
        mlContext.Transforms.Text.RemoveDefaultStopWords(
            outputColumnName: "WordsNoDefaultStop",
            inputColumnName: "Words",
            language: StopWordsRemovingEstimator.Language.Spanish
        )
    )
    // Stop words pesonalizadas
    .Append(
        mlContext.Transforms.Text.RemoveStopWords(
            outputColumnName: "WordsNoStop",
            inputColumnName: "WordsNoDefaultStop",
            stopwords: domainStop
        )
    )
    // Coneversion de los Tokens para los Ngrams
    .Append(
        mlContext.Transforms.Conversion.MapValueToKey(
            outputColumnName: "WordKeys",
            inputColumnName: "WordsNoStop"
        )
    )
    // Ngrams, utilizo un ngramLength de 2 para trabajar con secuencias
    // de dos palabras (bigram), y el useAllLengths a true para incluir secuencias
    // de una sola palabra (unigram)
    .Append(
        mlContext.Transforms.Text.ProduceNgrams(
            outputColumnName: "Ngrams",
            inputColumnName: "WordKeys",
            ngramLength: 2,
            useAllLengths: true,
            weighting: NgramExtractingEstimator.WeightingCriteria.Tf
        )
    )
    // Eligo un numberOfTopics de 6 para separar bien los temas,
    // ya que con un numero mas bajo el modelo mezcla temas,
    // aumento el maximumTokenCountPerDocument por los 500 registros 
    // que se manejan, y el numberOfSummaryTermsPerTopic de 8 para 
    // interpretar mejor los resultados obtenidos
    .Append(
        mlContext.Transforms.Text.LatentDirichletAllocation(
            outputColumnName: "Features",
            inputColumnName: "Ngrams",
            numberOfTopics: 6,
            alphaSum: 0.01f,
            beta: 0.01f,
            samplingStepCount: 4,
            maximumNumberOfIterations: 5000,
            maximumTokenCountPerDocument: 120,
            numberOfSummaryTermsPerTopic: 8
        )
    // Primera Version 
    // .Append(
    //     mlContext.Transforms.Text.LatentDirichletAllocation(
    //         outputColumnName: "Features",
    //         inputColumnName: "Ngrams",
    //         numberOfTopics: 4,
    //         alphaSum: 0.01f,
    //         beta: 0.01f,
    //         samplingStepCount: 4,
    //         maximumNumberOfIterations: 5000,
    //         maximumTokenCountPerDocument: 30,
    //         numberOfSummaryTermsPerTopic: 5
    //     )
    // )
    );

// Entrenamiento
var transformer = pipeline.Fit(dataview);

// Para obtener que palabras asocia a cada uno de los topicos y poder interpretar
// mejor los resultados obtenidos
var ldaTransformer = transformer.OfType<LatentDirichletAllocationTransformer>().Single();
var ldaDetails = ldaTransformer.GetLdaDetails(0);

Console.WriteLine();
Console.WriteLine("===== TOP PALABRAS POR TOPIC =====");

for (int t = 0; t < ldaDetails.WordScoresPerTopic.Count; t++)
{
    var topWords = ldaDetails.WordScoresPerTopic[t]
        .Select(ws => $"{ws.Word}({ws.Score:F3})");

    Console.WriteLine($"Topic{t + 1}: {string.Join(", ", topWords)}");
}

// Motor de predicciones
var predictionEngine = mlContext.Model.CreatePredictionEngine<TextoData, TextoDataTransformado>(transformer);

Console.WriteLine("==============================================");
Console.WriteLine("Topic1  Topic2  Topic3  Topic4  Topic5  Topic6");
PrintLdaFeatures(predictionEngine.Predict(ejemplos[5]));
PrintLdaFeatures(predictionEngine.Predict(ejemplos[35]));
PrintLdaFeatures(predictionEngine.Predict(ejemplos[65]));
PrintLdaFeatures(predictionEngine.Predict(ejemplos[95]));
PrintLdaFeatures(predictionEngine.Predict(ejemplos[145]));
PrintLdaFeatures(predictionEngine.Predict(ejemplos[195]));

Console.WriteLine();
Console.WriteLine("==============================================");
Console.WriteLine("Topic1  Topic2  Topic3  Topic4  Topic5  Topic6");
PrintLdaFeatures(predictionEngine.Predict(new() { Texto = "El paquete llegó tarde y la caja venía golpeada." }));
PrintLdaFeatures(predictionEngine.Predict(new() { Texto = "La calidad del producto es mala, se rompió en una semana." }));
PrintLdaFeatures(predictionEngine.Predict(new() { Texto = "Para el precio que tiene, esperaba mejores materiales." }));
PrintLdaFeatures(predictionEngine.Predict(new() { Texto = "Pedí devolución y el reembolso tardó más de una semana." }));

static void PrintLdaFeatures(TextoDataTransformado prediction)
{
    for (int i = 0; i < prediction.Features.Length; i++)
    {
        Console.Write($"{prediction.Features[i]:F4}  ");
    }
    Console.WriteLine($"{prediction.Texto}  ");
}