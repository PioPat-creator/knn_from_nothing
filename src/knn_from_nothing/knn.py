import math


# Najprostszy KNN - wszystko w jednej funkcji
def knn(punkt, dane, etykiety, k=3):
    """
    punkt: [x, y] - punkt do sklasyfikowania
    dane: lista punktów treningowych [[x1,y1], [x2,y2], ...]
    etykiety: lista etykiet ['A','B',...]
    k: liczba sąsiadów
    """
    # 1. Oblicz odległości
    odleglosci = []
    for i in range(len(dane)):
        dx = punkt[0] - dane[i][0]
        dy = punkt[1] - dane[i][1]
        odl = math.sqrt(dx * dx + dy * dy)  # euklidesowa
        odleglosci.append((odl, etykiety[i]))

    # 2. Sortuj od najmniejszej
    odleglosci.sort()

    # 3. Weź k najbliższych
    najblizsi = odleglosci[:k]

    # 4. Policz głosy
    glosy = {}
    for _, etykieta in najblizsi:
        glosy[etykieta] = glosy.get(etykieta, 0) + 1

    # 5. Znajdź zwycięzcę
    return max(glosy, key=glosy.get)


# PRZYKŁAD
if __name__ == "__main__":
    # Bardzo proste dane
    # Na lewo: owoce (F), na prawo: warzywa (V)
    punkty = [
        [1, 3], [2, 2], [1, 1],  # F
        [5, 5], [6, 4], [4, 6]  # V
    ]

    nazwy = ['F', 'F', 'F', 'V', 'V', 'V']

    print("KNN w 30 linijkach kodu!")
    print("-" * 30)

    # Test
    test_punkt = [3, 3]  # na środku
    wynik = knn(test_punkt, punkty, nazwy, k=3)
    print(f"Punkt {test_punkt} to: {wynij}")
    print(f"  (F = owoc, V = warzywo)")

    # Kilka testów
    print("\nWięcej testów:")
    testy = [[2, 3], [5, 5], [1, 2], [6, 5]]
    for p in testy:
        w = knn(p, punkty, nazwy, k=3)
        print(f"  {p} -> {w}")