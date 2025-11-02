using System;
namespace Ejercicio_04
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Introduce un número: ");
            int numero1 = 0;
            bool numeroValido = false;
            while (!numeroValido)
            {
                numeroValido = int.TryParse(Console.ReadLine(), out numero1);
                if (!numeroValido)
                {
                    Console.WriteLine("Introduce un número válido: ");
                }
            }
            numeroValido = false;
            Console.WriteLine("Introduce un número: ");
            int numero2 = 0;
            while (!numeroValido)
            {
                numeroValido = int.TryParse(Console.ReadLine(), out numero2);
                if (!numeroValido)
                {
                    Console.WriteLine("Introduce un número válido: ");
                }
            }
            if (numero1 != numero2)
            {
                Console.WriteLine("Los números son diferentes.");
                if (numero1 > numero2)
                {
                    Console.WriteLine($"El número mayor es: {numero1}");
                }
                else
                {
                    Console.WriteLine($"El número mayor es: {numero2}");
                }
            }
            else
            {
                Console.WriteLine("Los números son iguales.");
            }
        }
    }
}