



Pacotes Necessários
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dados = pd.read_csv('/content/train.csv')

dados.head()

dados = dados.drop( ['Name','Ticket','Cabin','Embarked'], axis=1)

"""# Nova seção"""

dados.head()

dados=dados.set_index(['PassengerId'])
dados=dados.rename(columns={'Survived':'target'}, inplace= False)

dados.head()

dados.describe()

dados.describe(include=['O'])

dados['Sex_F']=np.where(dados['Sex']=='female',1,0)
dados['Pclass_1']=np.where(dados['Pclass']==1,1,0)
dados['Pclass_2']=np.where(dados['Pclass']==2,1,0)
dados['Pclass_3']=np.where(dados['Pclass']==3,1,0)

dados.head()

dados.isnull().sum()

dados.fillna(0,inplace=True)
