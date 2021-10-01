import os
from model import models
def pickl(depth,tree):
        models(depth,tree)
        with open(os.path.join(os.path.dirname(__file__),'DTC_model.pkl'), 'wb') as file:
            pickle.dump(dtc, file)
            
        with open(os.path.join(os.path.dirname(__file__),'RFC_model.pkl'), 'wb') as file:
            pickle.dump(rfc, file)
            
        with open(os.path.join(os.path.dirname(__file__),'Scaler.pkl'), 'wb') as file:
            pickle.dump(sc, file)
       