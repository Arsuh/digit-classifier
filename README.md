# Digit-Classifier


# Prezentare Generală
Programul este scris în Python 3 și folosește bibliotecile Keras, Tensorflow și Numpy pentru a crea și folosi ceea ce se numește o rețea neuronală convolutivă (Convolutional Neural Network) care clasifică cifre scrise de mână.  Algoritmul folosește o bază de date de 10.000 de imagini alb-negru (MNIST dataset) pentru a găsi o generalizare aplicabilă datelor noi, reușind să identifice caracteristicile principale ale cifrelor și, chiar daca toată lumea scrie diferit, să le recunoască cu o acuratețe de peste 99%. Procesul are loc în două etape:

- Extragerea caracteristicilor, în care are loc preprocesarea imaginilor și ulterior crearea unor mape de caracteristici (modelul folosit repeta acest proces de 3 ori și creează mape cu 32, 64 și respectiv 128 de caracteristici)
- Clasificarea, în care ultima mapă de caracteristici este transformată în date de intrare pentru rețeaua neuronală propriu-zisă și rezultatul final este generat în ultimul strat de neuroni din rețea

# Descrieire fișiere
- Fișierul *Documentatie Digit Classifier.docx* este documentația proiectului
- Fișierele *CNN_mnist.py*, *CNN_mnist_best.py* și *CNN_mnist_best_improve.py* conțin algoritmi care încearcă să creeze o rețea neuronală cu o acuratețe cât mai mare, modificând parametrii acesteia.
- În folder-ul *best-models/* se găsește o colecție de modele antrenate, performanțele fiecăruia găsindu-se într-un fișier de tip *.txt*. Rețelele neuronale cele mai bune se află în folder-ul *single_models/*, iar cel cu acuratețea cea mai mare este *Best.hdf5*
- Cu ajutorul fisierului *plot.py* se poate reprezenta grafic evoluția fiecărui model
- În folder-ul *App/* se găsește aplicația propriu-zisă cu numele *app.py*. Pentru a crea baza de date secundară a fost folosit fișierul *training_app.py*. În branch-ul *windows_releases* se pot găsi și executabilele acestot fișiere.

# Instalare
Pentru rularea aplicației este necesară orice versiune de Python 3 pe 64 de biți. Instalarea bibliotecilor necesare (Tensorflow, Keras, Numpy, PIL, Tkinter și Matplotlib) se poate realiza din consola sistemului de operare cu ajutorul comenzii “pip install”. Aplicația suportă atât versiunea pentru procesor a bibliotecii Tensorflow, cât și cea pentru placa video.
