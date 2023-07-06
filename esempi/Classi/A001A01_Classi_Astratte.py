from abc import ABC, abstractmethod

'''
Le classi astratte sono utili quando si desidera definire 
un'interfaccia comune per un gruppo di classi correlate, 
senza fornire un'implementazione concreta. Le classi astratte 
consentono di stabilire un contratto che le sottoclassi 
devono seguire, garantendo così una struttura coerente del codice.
'''

class Forma(ABC):
    @abstractmethod
    def calcola_area(self):
        pass

    @abstractmethod
    def calcola_perimetro(self):
        pass
    
    
class Rettangolo(Forma):
    def __init__(self, lunghezza, larghezza):
        self.lunghezza = lunghezza
        self.larghezza = larghezza

    def calcola_area(self):
        return self.lunghezza * self.larghezza

    def calcola_perimetro(self):
        return 2 * (self.lunghezza + self.larghezza)
    

ll = Rettangolo(5,3)
print(ll.calcola_area())
# se voglio vedere se la classe è stata istanziata
print(ll)