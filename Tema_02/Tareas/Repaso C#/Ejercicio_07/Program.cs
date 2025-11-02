using System;

namespace Ejercicio_07
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Mascota mascota1 = new Mascota("Perro", 2);
            Mascota mascota2 = new Mascota("Gato", 3);

            Console.WriteLine("--- MASCOTAS ---");
            mascota1.MostrarInfo();
            mascota2.MostrarInfo();

            Persona persona1 = new Persona("Sergio", "Orgaz", "Bravo", 22, mascota1);
            Persona persona2 = new Persona("Ana", "López", "Martínez", 25, mascota2);

            Console.WriteLine("--- PERSONAS ---");
            persona1.MostrarInfo();
            persona2.MostrarInfo();
        }
    }
}
