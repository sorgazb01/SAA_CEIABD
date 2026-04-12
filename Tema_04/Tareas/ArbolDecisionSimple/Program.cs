// Matriz de datos
var datos = new (bool Estudia, bool DuermeBien, bool EntregaTareas, bool Aprueba)[]
{
    (true,  true,  true,  true),
    (true,  false, true,  true),
    (false, true,  true,  true),
    (false, false, false, false),
    (true,  true,  false, true),
    (false, false, true,  false),
};

// Metodo para predecir si el alumno aprueba o no
static bool Predecir(bool estudia, bool duermeBien, bool entregaTareas)
{
    if (estudia)
    {
        return true;
    }
    else
    {
        if (duermeBien)
        {
            return true;
        }
        else
        {
            return false; 
        }
    }
}

// Casos ejercicio
foreach (var (estudia, duermeBien, entregaTareas, aprueba) in datos)
{
    bool prediccion = Predecir(estudia, duermeBien, entregaTareas);
    string apruebaResultado;
    if (prediccion == aprueba)
    {
        apruebaResultado = "Aprueba";
    }
        
    else
    {
        apruebaResultado = "No aprueba";
    }
    Console.WriteLine($"Estudia={estudia} DuermeBien={duermeBien} EntregaTareas={entregaTareas} → Predicción={prediccion} (Real={aprueba}) {apruebaResultado}");
}

// Casos nuevos
Console.WriteLine("Casos nuevos");

bool caso1 = Predecir(false, false, true);
if (caso1)
{
    Console.WriteLine("(Estudia=No, DuermeBien=No, EntregaTareas=Sí)  → Aprueba");
}
else
{
    Console.WriteLine("(Estudia=No, DuermeBien=No, EntregaTareas=Sí)  → No aprueba");
}

bool caso2 = Predecir(true, false, false);
if (caso2)
{
    Console.WriteLine("(Estudia=Sí, DuermeBien=No, EntregaTareas=No)  → Aprueba");    
}
else
{
    Console.WriteLine("(Estudia=Sí, DuermeBien=No, EntregaTareas=No)  → No aprueba");
}

bool caso3 = Predecir(false, true, false);
if (caso3)
{
    Console.WriteLine("(Estudia=No, DuermeBien=Sí, EntregaTareas=No)  → Aprueba");
}
else
{
    Console.WriteLine("(Estudia=No, DuermeBien=Sí, EntregaTareas=No)  → No aprueba");
}