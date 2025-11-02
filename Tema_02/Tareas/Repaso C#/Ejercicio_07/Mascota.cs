public class Mascota
{
    public string Nombre { get; set; }
    public int Edad { get; set; }

    public Mascota(string nombre, int edad)
    {
        Nombre = nombre;
        Edad = edad;
    }

    public void MostrarInfo()
    {
        Console.WriteLine($"Nombre: {Nombre}, Edad: {Edad} años");
    }
}
