# cubic-sphere

Simulare un random walk su sfera quadridimensionale per ricostruire il rumore di un giroscopio.

## Esecuzione

g++ -std::c++23 mainGrid4.cpp grid4.cpp function.cpp surface.cpp -Ofast -ltbb

## Idea

Il giroscopio ha del rumore che porta a una fallace orientazione.
Usando i quaternioni, l'integrazione del segnale può essere visto come il movimento di un punto su una sfera quadridimensionale unitaria.
In un segnale di solo rumore, se non è presente del drift, tale punto si muoverà di moto Browniano.

Siamo intenzionati a capire dopo quanto il segnale di un giroscopio perde di utilità perché il rumore potrebbe averlo portato ovunque sulla sfera.
Lo scopo è vedere come diffonde la densità di probabilità.

## Fokker-Planck

Per fare questo usiamo la Fokker-Planck per calcolare la diffusione di probabilità dell'orientazione.

Essa è fatta per evolvere a raggio costante, ma vista la discretizzazione si diffonde anche lungo il raggio.
Visto che il passo spaziale non può essere troppo piccolo per motivi computazionali, la diffusione lungo il raggio è eccessiva e occorrerebbe proiettare sulla sfera troppo di frequente.
Usando un coefficiente davanti alla PDE uguale a 1, con passo spaziale 0.1 il voxel è 1e-4.
Con passo temporale 1e-5 devo effettuare 1e5 step per arrivare a un tempo 1; tempo in cui la densità diffonde su tutta la sfera.

Dai dati, col passo temporale 1e-5 bisogna proiettare dopo 0.01.
Con 1e-4 la convergenza di Crank-Nicolson è troppo lenta e fallisce a volte.
Il problema è che servirebbe un passo spaziale più piccolo, ma in quattro dimensioni il numero di punti cresce velocemente e occorre lasciare spazio attorno alla sfera per evitare che la condizione assorbente al contorno agisca prima della proiezione.
