# Bachelorarbeit

Diese Bachelorarbeit beschäfftigt sich mit dem Vergelich verschiederen Baysischaner Optimierungsmethoden.
*Ziel:*

Also Input Paramter:
- Baysianischen Optimieren ( Suggaratemodelle) ?
- Anzahl an Dimensionen
- Metriken zum Maß für ähnlichkeiten von Funktionen
    -   Lp Distanzen
    -   MSE
    -   Cosine Similartity
    -   Wasserstein Distanz
    -   Bergmann Distanz
    -    



- https://www.cambridge.org/core/books/bayesian-optimization/11AED383B208E7F22A4CE1B5BCBADB44
- https://www.sciencedirect.com/science/article/pii/S0950705124008207
- https://botorch.org/
- https://arxiv.org/pdf/1807.02811
- https://coco-platform.org/testsuites/bbob/overview.html
- https://numbbo.github.io/coco-doc/apidocs/cocoex/
- https://github.com/numbbo/coco?tab=readme-ov-file
- https://publica.fraunhofer.de/entities/publication/8eea9127-4f42-4cb7-b6ef-4274e9d68ad9
- https://arxiv.org/abs/2410.24222






Ziel ist es einen Datframe zu erzuegen der für jede Testkombination die Messwerte speichert sodass dieser Dataframe mittel Pipes ananlysiert werden kann. 

*Workflow der Anwendung:*
1. Eingabe der zu Testen Funktion, Dimensionen, Surrogatemodele, Acquisitionsfunktionen
   sowie der Größe der Sampels und Anzahl an Itertion
2. Erstellen des df_kombination:
   Hier werden allen aus den oberen Input Information bis auf die Größe der Samples und     der Anzahl Iteration. Die Informatation werden Kombiiert und alle Kombinatione werden    als eien Varainet abspeichert
3. Starten der Schleife über die Kombination also die Zeilen des df_kombination
4. Erstellung der Testdaten
   4.1 Aufrufen der Suite aus dem Coco Package --> Suites mit Problemen
5. Für jeden Ober Problem werden die Messwerte pro Sample und Iteration berechnet
   und einzelen Dataframe erstellt  
7. Zusammenfügen der gewünschten Dataframe
8. Analysen erstellen
9. Plots erstellen 
