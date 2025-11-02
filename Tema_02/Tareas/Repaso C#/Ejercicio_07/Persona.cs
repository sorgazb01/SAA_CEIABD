public class Persona
{
    public string Nombre { get; set; }
    public string Apellido1 { get; set; }
    public string Apellido2 { get; set; }
    public int Edad { get; set; }
    public Mascota Mascota { get; set; }
    public Persona(string nombre, string apellido1, string apellido2, int edad, Mascota mascota)
    {
        Nombre = nombre;
        Apellido1 = apellido1;
        Apellido2 = apellido2;
        Edad = edad;
        Mascota = mascota;
    }
    public void MostrarInfo()
    {
        Console.WriteLine($"Nombre: {Nombre} {Apellido1} {Apellido2}, Edad: {Edad} años");
        Console.Write("Mascota: ");
        Mascota.MostrarInfo();
    }
}
