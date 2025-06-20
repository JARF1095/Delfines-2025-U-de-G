import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, 
                           precision_score, recall_score, 
                           f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import time
from deap import base, creator, tools, algorithms

# Cargar datos
df = pd.read_csv('data.csv')

# Preprocesamiento mejorado para evitar warnings
df['diagnosis'] = df['diagnosis'].map({'M':0, 'B':1}).astype(int)
df = df.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')

# Verificar y convertir datos
features = df.columns.difference(['diagnosis'])
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

X = df[features]
y = df['diagnosis']

# División de datos
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Configuración del Algoritmo Genético
def evalXGB(individual):
    """Función de evaluación para el algoritmo genético"""
    max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample, colsample_bytree = individual
    
    # Asegurar que gamma no sea negativo
    gamma = max(0, gamma)  # gamma (min_split_loss) debe ser >= 0
    
    params = {
        'max_depth': int(max_depth),
        'learning_rate': max(0.001, learning_rate),  # Asegurar learning_rate positivo
        'n_estimators': int(n_estimators),
        'gamma': gamma,
        'min_child_weight': int(min_child_weight),
        'subsample': np.clip(subsample, 0.5, 1.0),  # Mantener entre 0.5 y 1.0
        'colsample_bytree': np.clip(colsample_bytree, 0.5, 1.0),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
        # Eliminamos use_label_encoder ya que es obsoleto
    }
    
    try:
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        return (accuracy,)  # DEAP requiere que los fitness sean tuplas
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        return (0.0,)  # Devolver fitness mínimo si hay error

# Definir los tipos para el algoritmo genético
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Definir rangos de los parámetros con restricciones
toolbox.register("attr_max_depth", np.random.randint, 2, 9)
toolbox.register("attr_learning_rate", np.random.uniform, 0.01, 0.3)
toolbox.register("attr_n_estimators", np.random.randint, 50, 301)
toolbox.register("attr_gamma", np.random.uniform, 0, 1)  # Gamma ahora siempre >= 0
toolbox.register("attr_min_child_weight", np.random.randint, 1, 11)
toolbox.register("attr_subsample", np.random.uniform, 0.5, 1.0)
toolbox.register("attr_colsample_bytree", np.random.uniform, 0.5, 1.0)

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_max_depth, toolbox.attr_learning_rate, 
                  toolbox.attr_n_estimators, toolbox.attr_gamma, 
                  toolbox.attr_min_child_weight, toolbox.attr_subsample, 
                  toolbox.attr_colsample_bytree), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operadores genéticos mejorados
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=0.1, low=[2, 0.01, 50, 0, 1, 0.5, 0.5], 
                 up=[8, 0.3, 300, 1, 10, 1.0, 1.0])
toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.1, low=[2, 0.01, 50, 0, 1, 0.5, 0.5], 
                 up=[8, 0.3, 300, 1, 10, 1.0, 1.0], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalXGB)

# Configuración del algoritmo genético
population_size = 20
num_generations = 15
cx_prob = 0.7
mut_prob = 0.3

# Ejecutar la optimización
print("Iniciando optimización con Algoritmo Genético...")
start_time = time.time()

population = toolbox.population(n=population_size)
hof = tools.HallOfFame(1)  # Guardar el mejor individuo

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

# Agregar manejo de excepciones durante la evolución
try:
    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=cx_prob, mutpb=mut_prob,
        ngen=num_generations, stats=stats, halloffame=hof,
        verbose=True
    )
except Exception as e:
    print(f"Error durante la optimización: {e}")
    print("Intentando recuperar el mejor individuo encontrado hasta ahora...")
    if len(hof) > 0:
        best_individual = hof[0]
    else:
        print("No se encontraron individuos válidos. Usando parámetros por defecto.")
        best_individual = [6, 0.1, 100, 0, 1, 0.8, 0.8]

print(f"Optimización completada en {time.time()-start_time:.2f} segundos")

# Obtener el mejor individuo
best_individual = hof[0]
best_params = {
    'max_depth': int(best_individual[0]),
    'learning_rate': max(0.001, best_individual[1]),
    'n_estimators': int(best_individual[2]),
    'gamma': max(0, best_individual[3]),
    'min_child_weight': int(best_individual[4]),
    'subsample': np.clip(best_individual[5], 0.5, 1.0),
    'colsample_bytree': np.clip(best_individual[6], 0.5, 1.0),
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1
}

print("\nMejores parámetros encontrados:")
print(best_params)
print(f"\nMejor accuracy en validación: {best_individual.fitness.values[0]:.4f}")

# Entrenar el mejor modelo con todos los datos de entrenamiento
best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

# =============================================
# Evaluación del modelo (igual que en tu código original)
# =============================================

# Predicciones
train_probs = best_model.predict_proba(X_train)[:, 1]
test_probs = best_model.predict_proba(X_test)[:, 1]

train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

# Métricas de performance
metricsXGB = pd.DataFrame({
    'metric': ['AUC', 'Gini', 'Accuracy', 'Precision', 'Recall', 'F1-score'],
    'xgb_train': [
        roc_auc_score(y_train, train_probs),
        (roc_auc_score(y_train, train_probs)*2-1),
        accuracy_score(y_train, train_preds),
        precision_score(y_train, train_preds),
        recall_score(y_train, train_preds),
        f1_score(y_train, train_preds)
    ],
    'xgb_test': [
        roc_auc_score(y_test, test_probs),
        (roc_auc_score(y_test, test_probs)*2-1),
        accuracy_score(y_test, test_preds),
        precision_score(y_test, test_preds),
        recall_score(y_test, test_preds),
        f1_score(y_test, test_preds)
    ]
})

print("\nMétricas de performance:")
print(metricsXGB)

# Specificity
tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
specificity = tn/(tn+fp)
print('\nTest Specificity:', specificity)

# Importancia de las variables
importance = best_model.get_booster().get_score(importance_type='total_gain')
pdVarImp = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Importance': list(importance.values())
}).sort_values('Importance', ascending=False)

pdVarImp['Orden'] = np.arange(len(pdVarImp)) + 1
pdVarImp['porc_gain'] = pdVarImp.Importance.apply(lambda x: x/pdVarImp.Importance.sum())
pdVarImp['porc_gain_acum'] = pdVarImp.porc_gain.cumsum(axis=0)

print("\nTop 10 variables más importantes:")
print(pdVarImp.head(10))

# Parámetros finales del modelo
print("\nParámetros finales del modelo:")
print(best_model.get_params())

# Gráfico de evolución del algoritmo genético
gen = logbook.select("gen")
avg = logbook.select("avg")
min_ = logbook.select("min")
max_ = logbook.select("max")

plt.figure(figsize=(10, 6))
plt.plot(gen, avg, label="Promedio")
plt.plot(gen, min_, label="Mínimo")
plt.plot(gen, max_, label="Máximo")
plt.xlabel("Generación")
plt.ylabel("Accuracy")
plt.title("Evolución del Fitness (Accuracy) en el Algoritmo Genético")
plt.legend()
plt.show()