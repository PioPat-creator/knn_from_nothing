class KNNClassifier:
    def __init__(self, k=3, metric="euclidean"):
        import numpy as np

        # 1. Funkcja odległości euklidesowej
        def odleglosc(p1, p2):
            """
            Oblicza odległość między dwoma punktami
            p1, p2: listy lub tablice [x, y]
            """
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        # 2. Najprostszy KNN w jednej funkcji
        def prosty_knn(punkt_testowy, punkty_treningowe, etykiety, k=3):
            """
            Przewiduje klasę dla punktu testowego

            punkt_testowy: [x, y]
            punkty_treningowe: lista punktów [[x1, y1], [x2, y2], ...]
            etykiety: lista etykiet dla każdego punktu treningowego
            k: liczba sąsiadów
            """
            # Oblicz wszystkie odległości
            odleglosci = []
            for i, punkt in enumerate(punkty_treningowe):
                d = odleglosc(punkt_testowy, punkt)
                odleglosci.append((d, etykiety[i]))

            # Sortuj od najmniejszej odległości
            odleglosci.sort(key=lambda x: x[0])

            # Weź k najbliższych
            najblizsi = odleglosci[:k]

            # Zlicz głosy
            glosy = {}
            for _, etykieta in najblizsi:
                if etykieta not in glosy:
                    glosy[etykieta] = 0
                glosy[etykieta] += 1

            # Znajdź zwycięzcę
            return max(glosy.items(), key=lambda x: x[1])[0]

        # PRZYKŁAD
        if __name__ == "__main__":
            print("NAJPROSTSZY PRZYKŁAD KNN")
            print("=" * 40)

            # Dane treningowe - prosty przykład
            # 3 czerwone (R) i 3 niebieskie (B) punkty
            dane = [
                [1, 2],  # R
                [2, 1],  # R
                [1, 1],  # R
                [4, 5],  # B
                [5, 4],  # B
                [5, 5]  # B
            ]

            kolory = ['R', 'R', 'R', 'B', 'B', 'B']

            # Punkty do przetestowania
            testowe = [
                [1.5, 1.5],  # powinien być R
                [4.5, 4.5],  # powinien być B
                [3, 3]  # na środku - ciekawe co wybierze
            ]

            print("Dane treningowe:")
            for i, (punkt, kolor) in enumerate(zip(dane, kolory), 1):
                print(f"  Punkt {i}: {punkt} -> {kolor}")

            print("\nTest dla k=3:")
            for punkt in testowe:
                wynik = prosty_knn(punkt, dane, kolory, k=3)
                print(f"  Punkt {punkt} -> Klasa: {wynik}")

            print("\nSprawdźmy różne wartości k dla punktu [3, 3]:")
            punkt_sporny = [3, 3]
            for k in [1, 3, 5]:
                wynik = prosty_knn(punkt_sporny, dane, kolory, k=k)
                print(f"  k={k}: -> {wynik}")

            # Wizualizacja w ASCII
            print("\nWizualizacja:")
            print("  R - czerwone punkty (lewy dolny róg)")
            print("  B - niebieskie punkty (prawy górny róg)")
            print("  ? - punkty testowe")

            # Prosta siatka
            print("\n  y")
            print("  5 B  B  ?  B  B")
            print("  4 .  .  .  .  B")
            print("  3 .  .  ?  .  .")
            print("  2 R  .  .  .  .")
            print("  1 R  R  .  .  .")
            print("    1  2  3  4  5  x")
    def fit(self, X, y):
        ...
    def predict(self, X):
        ...
