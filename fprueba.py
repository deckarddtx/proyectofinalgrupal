import streamlit as st
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt


#from streamlit_option_menu import option_menu

#with st.sidebar:
    #selected = option_menu("Main Menu", ["MARCO TEORICO", 'LECTURA DE DATOS', 'MAPA DE CORRELACION', 'MAPA DE CALOR','PROPUESTA','VALIDACION','CORRELACION','MAXIMOS'], 
        #icons=['house', 'gear'], menu_icon="cast", default_index=1)
    #selected
    
st.write("""PROYECTO FINAL""")
st.write("""Universidad Nacional de San Agustín de Arequipa
Escuela Profesional de Ingeniería de Telecomunicaciones""")
from PIL import Image
image = Image.open('unsa-logo.png')

st.image(image, caption='Sunrise by the mountains')

st.write("""Ingeniero Renzo Bolivar - Docente DAIE
Curso : Computación 1""")


st.write("""GRUPO A - GRUPO 2""")
st.write("""Alumnos:""") 
st.write("""ABRIL MESTAS RAUL EDILBERTO""")
st.write("""ALMIRON PATIÑO ROSA LINDA""")
st.write("""CCOA CUYO PAULA MARCIA""")
st.write("""PAREDES MORON ROSARIO ISABEL""") 

st.write("""
INVESTIGACIÓN FORMATIVA
PROYECTO FINAL
PYTHON - Inteligencia Artificial


OBJETIVOS
Los Objetivos de la investigación formativa son:

Competencia Comunicativa Presentación de sus resultados con lenguaje de programación Python utilizando los archivos Jupyter Notebook.
Competencia Aprendizaje: con las aptitudes en Descomposición (desarticular el problema en pequeñas series de soluciones), Reconocimiento de Patrones (encontrar simulitud al momento de resolver problemas), Abstracción (omitir información relevante), Algoritmos (pasos para resolución de un problema).
Competencia de Trabajo en Equipo: exige habilidades individuales y grupales orientadas a la cooperación, planificación, coordinación, asignación de tareas, cumplimiento de tareas y solución de conflictos en pro de un trabajo colectivo, utilizando los archivos Jupyter Notebook los cuales se sincronizan en el servidor Gitlab con comandos Git.


Aplicación en IA
Sistema Recomendador


El Sistema recomendador deberá encontrar la compatibilidad o similitud entre un grupo de personas encuestadas, en las áreas de:
-Musica

-Peliculas

-Comida

-Lugares que desean Conocer

-Obras de Arte

La compatibilidad o similitud será encontrada con el algoritmo de Correlación de Pearson y será verificada con la La Matrix de Correlación de Pearson con una librería de Python y utilizando una función personal

Base Teórica
Análisis de Correlación
A menudo nos interesa observar y medir la relación entre 2 variables numéricas mediante el análisis de correlación. Se trata de una de las técnicas más habituales en análisis de datos y el primer paso necesario antes de construir cualquier modelo explicativo o predictivo más complejo. Para poder tener el Datset hay que recolectar información a travez de encuentas.

¿Qué es la correlación?
La correlación es un tipo de asociación entre dos variables numéricas, específicamente evalúa la tendencia (creciente o decreciente) en los datos.

Dos variables están asociadas cuando una variable nos da información acerca de la otra. Por el contrario, cuando no existe asociación, el aumento o disminución de una variable no nos dice nada sobre el comportamiento de la otra variable.

Dos variables se correlacionan cuando muestran una tendencia creciente o decreciente.

¿Cómo se mide la correlación?
Tenemos el coeficiente de correlación lineal de Pearson que se sirve para cuantificar tendencias lineales, y el coeficiente de correlación de Spearman que se utiliza para tendencias de aumento o disminución, no necesariamente lineales pero sí monótonas.

Correlación de Pearson
El coeficiente de correlación lineal de Pearson mide una tendencia lineal entre dos variables numéricas.
Es el método de correlación más utilizado, pero asume que:

La tendencia debe ser de tipo lineal.
No existen valores atípicos (outliers).
Las variables deben ser numéricas.
Tenemos suficientes datos (algunos autores recomiendan tener más de 30 puntos u observaciones).
Los dos primeros supuestos se pueden evaluar simplemente con un diagrama de dispersión, mientras que para los últimos basta con mirar los datos y evaluar el diseño que tenemos.

Cómo se interpreta la correlación
El signo nos indica la dirección de la relación, como hemos visto en el diagrama de dispersión.

un valor positivo indica una relación directa o positiva,
un valor negativo indica relación indirecta, inversa o negativa,
un valor nulo indica que no existe una tendencia entre ambas variables (puede ocurrir que no exista relación o que la relación sea más compleja que una tendencia, por ejemplo, una relación en forma de U).
La magnitud nos indica la fuerza de la relación, y toma valores entre −1 a 1 . Cuanto más cercano sea el valor a los extremos del intervalo ( 1 o −1 ) más fuerte será la tendencia de las variables, o será menor la dispersión que existe en los puntos alrededor de dicha tendencia. Cuanto más cerca del cero esté el coeficiente de correlación, más débil será la tendencia, es decir, habrá más dispersión en la nube de puntos.

si la correlación vale 1 o −1 diremos que la correlación es “perfecta”, si la correlación vale 0 diremos que las variables no están correlacionadas.""")


st.write("""MARCO TEORICO""")
st.write(""" ¿Que es machine learning?
Machine learning: Es una rama de la inteligencia artificial que es autónoma ya que puede realizar predicciones a partir de procesamiento de datos. Esta tecnología está presente en un sinfín de aplicaciones como las recomendaciones de Netflix o Spotify, las respuestas inteligentes de Gmail o el habla de Siri y Alexa. El ‘machine learning’ es un maestro del reconocimiento de patrones, y es capaz de convertir una muestra de datos en un programa informático capaz de extraer inferencias de nuevos conjuntos de datos para los que no ha sido entrenado previamente Aunque ahora esté de moda, gracias a su capacidad para derrotar a jugadores del Go o resolver cubos de Rubik, su origen se remonta al siglo pasado. “La estadística es sin duda la base fundamental del aprendizaje automático, que básicamente consiste en una serie de algoritmos capaces de analizar grandes cantidades de datos para deducir cuál es el resultado más óptimo para un determinado problema”, añade Espinoza.

""")

st.write("""LIBRERIAS PARA CIENCIA DE DATOS:¶
Conjuntos de archivos de código que han sido creados para desarrollar software de manera sencilla. Gracias a ellas, los desarrolladores pueden evitar la duplicidad de código y minimizar errores con mayor agilidad .Estas librerías son altamente prácticas a la hora de implementar flujos de Machine Learning.

librerías para ciencia de datos: Conjuntos de archivos de código que han sido creados para desarrollar software de manera sencilla. Gracias a ellas, los desarrolladores pueden evitar la duplicidad de código y minimizar errores con mayor agilidad .Estas librerías son altamente prácticas a la hora de implementar flujos de Machine Learning.""")
st.write("""¿Que es NUMPY?
Es una librería de Python especializada en el cálculo numérico y el análisis de datos, especialmente para un gran volumen de datos.Incorpora arrays lo que permite representar colecciones de datos de un mismo tipo en varias dimensiones, y funciones muy eficientes para su manipulación.Además cuenta con múltiples herramientas para manejar matrices de una forma muy eficiente.

El módulo Numpy introduce en escena un nuevo tipo de objeto, ndarray (n dimensional array), caracterizado por:

Almacenamiento eficiente de colecciones de datos del mismo tipo
Conjunto de métodos que permiten operar de forma vectorizada sobre sus datos
Las formas más habituales de crear un nuevo array son:

A partir de otras colecciones de datos de Python, como listas o tuplas
Desde cero mediante funciones específicas
Leyendo los datos de un fichero""")
st.write("""¿Qué es Pandas?
Pandas es una librería de código abierto que ofrece unas estructuras muy poderosas y flexibles que facilitan la manipulación y tratamiento de datos.Las principales funciones de pandas son :cargar datos, modelar, analizar, manipular y prepararlos.

Estructuras de datos Pandas:
Cuenta con dos estructuras estas son:

Series: array unidimensional etiquetado capaz de almacenar cualquier tipo de dato.
DataFrame: estructura bidimensional con columnas que pueden ser también de cualquier tipo. Estas columnas son a su vez Series.
¿Como analizar datos con Pandas?
head(n): Esta función devuelve las primeras n filas de nuestro DataFrame.
tail(n): Devuelve las n últimas filas de nuestro DataFrame.
describe(): Esta función da estadísticas descriptivas incluyendo aquellas que resumen la tendencia central, dispersión y la forma de la distribución de los datos.""")

st.write("""¿Qué es matplotlib? Es una biblioteca para la generación de gráficos en dos dimensiones, a partir de datos contenidos en listas en el lenguaje de programación Python. Permite crear y personalizar los tipos de gráficos más comunes, entre ellos:

Diagramas de barras
Histograma
Diagramas de sectores
Diagramas de caja y bigotes
Diagramas de violín
Diagramas de dispersión o puntos
Diagramas de lineas
Diagramas de areas
Diagramas de contorno
Mapas de color Mapas de color y combinaciones de todos ellos.""")

st.write("""Seaborn:
Es una librería para Python que permite generar fácilmente gráficos. Seaborn está basada en matplotlib y proporciona una interfaz de alto nivel . Tiene como objetivo convertir la visualización en una parte central de la exploración y comprensión de los datos, generando atractivas gráficas con sencillas funciones que ofrecen una interfaz semejante, facilitando el paso de unas funciones a otras.""")
st.write("""¿Qué es la escala Likert?
Se utiliza para medir qué tan de acuerdo están los encuestados con una variedad de afirmaciones.Esta es ideal para medir reacciones, actitudes y comportamientos de una persona. A diferencia de una simple pregunta de “sí” / “no”, la escala de Likert permite a los encuestados calificar sus respuestas.

La escala de Likert es uno de los tipos de escalas de medición utilizados principalmente en la investigación de mercados para la comprensión de las opiniones y actitudes de un consumidor hacia una marca, producto o mercado meta. Nos sirve principalmente para realizar mediciones y conocer sobre el grado de conformidad de una persona o encuestado hacia determinada oración afirmativa o negativa.

Las respuestas pueden ser ofrecidas en diferentes niveles de medición, permitiendo escalas de 5, 7 y 9 elementos configurados previamente. Siempre se debe tener un elemento neutral para aquellos usuarios que ni de acuerdo ni en desacuerdo.""")

st.write("""CSV (Valores Separados por Comas):
Es el formato más común de importación y exportación de hojas de cálculo y bases de datos. Es cualquier archivo de texto en el cual los caracteres están separados por comas, haciendo una especie de tabla en filas y columnas. Las columnas quedan definidas por cada punto y coma (;), mientras que cada fila se define mediante una línea adicional en el texto. De esta manera, se pueden crear archivos CSV con gran facilidad.

¿Para qué sirve un archivo CSV?
Los archivos CSV sirven para manejar una gran cantidad de datos en formato tabla, sin que ello conlleve sobrecoste computacional alguno. Es tremendamente sencillo generar una tabla a partir de un documento de texto, con tan solo delimitar cada celda requerida con un punto y coma o con una coma).""")
st.write("""Gráfica de calor
Un gráfico de calor se usa para visualizar la relación numérica existente entre dos variables de categorías.Esta consiste en una cuadrícula rectangular compuesta de dos variables de categorías,cada celda de la cuadrícula se simboliza con un valor numérico. tiene limitaciones? Las variables de un gráfico de calor no pueden tener más de 3.000 valores únicos por eje. Si una de las variables o ambas exceden el límite de 3.000 valores, puede usarse un filtro, por ejemplo, un filtro predefinido, para reducir el tamaño del dataset.

Matriz de correlación:
Es una tabla que indica los coeficientes de conexión entre los factores,se utiliza para bosquejar información, como contribución a una investigación más desarrollada. Normalmente, un marco de relación es “cuadrado”, con factores similares que aparecieron en las líneas y secciones. Aplicaciones de una matriz de correlación medición de la relación La mayoría de los marcos de relación utilizan la Conexión de Minuto de Artículo de Pearson (r). Codificación de los factores En el caso de que también tengas información de un resumen, tendrás que elegir cómo codificar la información antes de procesar las conexiones. Tratamiento de las cualidades que faltan La información que utilizamos para procesar las conexiones a menudo contiene cualidades que faltan. Esto puede deberse a que no hemos recogido esta información o a que no tenemos la menor idea de las reacciones. Existen diferentes procedimientos para manejar las cualidades perdidas cuando se procesan las redes de conexión. La mejor práctica es, en su mayor parte, utilizar numerosas atribuciones.

""")
st.write("""Ques es un Dashboard?
Es una herramienta personalizable de visualización de datos, que te ayuda a conectar tus archivos, servicios, API o archivos adjuntos, y muestra estos datos como tablas, tipos de gráficas u otras visualizaciones de datos al espectador y reduce el esfuerzo manual. tipos de dashboard

¿Cómo funcionan los dashboards?
Los dashboards responden a preguntas importantes sobre tu negocio. A diferencia de las herramientas avanzadas de inteligencia empresarial, los dashboards están diseñados para el análisis rápido y el conocimiento de la información. El enfoque más común para diseñar un dashboard es construirlo utilizando un formato de pregunta-respuesta.

DATOS FALTANTES
Son aquellos que no constan debido a cualquier acontecimiento, como por ejemplo errores en la transcripción de los datos o la ausencia de disposición a responder a ciertas cuestiones de una encuesta. Los datos pueden faltar de manera aleatoria o no aleatoria.

Los datos faltantes aleatorios pueden perturbar el análisis de datos dado que disminuyen el tamaño de las muestras y en consecuencia la potencia de las pruebas de contraste de hipótesis. Los datos faltantes no aleatorios ocasionan, además, disminución de la representatividad de la muestra.

Tratamiento

De casos completos o eliminación por lista
Este procedimiento consiste en incluir en el análisis los casos que presentan observaciones completas en todas las variables. Este método solo debe utilizarse cuando el proceso de recogida de datos es aleatorio, porque en otro caso introduce sesgo. Otro inconveniente es que el tamaño muestral puede llegar a sufrir una gran reducción y afectar a la representatividad de la muestra.

Selección por variables

Se mantienen en la base de datos los casos con tal que tengan datos en las variables que van a ser utilizadas para el análisis. Este procedimiento tiene el inconveniente de generar muestras heterogéneas.

Métodos de imputación
Los métodos de imputación consisten en estimar los valores ausentes en base a los valores válidos de otras variables y/o casos de la muestra. La estimación se puede hacer a partir de la información del conjunto completo de variables o bien de algunas variables especialmente seleccionadas. Usualmente los métodos de imputación se utilizan con variables métricas (de intervalo o de razón), y deben aplicarse con gran precaución porque pueden introducir relaciones inexistentes en los datos realas.

Principales procedimientos:
Sustitución por la Media. Consiste en sustituir el valor ausente por la Media de los valores válidos. Este procedimiento plantea inconvenientes como:
Dificulta la estimación de la Variáncia.

Distorsiona la verdadera distribución de la variable,

Distorsiona la correlación entre variables dado que añade valores constantes.

Sustitución por constante. Consiste en sustituir los valores ausentes por constantes cuyo valor viene determinado por razones teóricas o relacionadas con la investigación previa. Presenta los mismos inconvenientes que la sustitución por la Media, y solo debe ser utilizado si hay razones para suponer que es más adecuado que el método de la media.

Imputación por regresión. Este método consiste en estimar los valores ausentes en base a su relación con otros variables mediante Análisis de Regresión.

Inconvenientes:

Incrementa artificialmente las relaciones entre variables.

Hace que se subestime la Variáncia de las distribuciones.

Asume que las variables con datos ausentes tienen relación de alta magnitud con las otras variables.

¿Qué es un framework?
Es una especie de plantilla, un esquema conceptual, que simplifica la elaboración de una tarea, ya que solo es necesario complementarlo de acuerdo a lo que se quiere realizar.""")


st.write("""Propuesta
linea 2

1.- Dataset
Formulario de Google (Preguntas)
Por Ustedes
ME VAS A EXTRAÑAR
TE VI VENIR
MARC ANTHONY -VIVIR MI VIDA
LOS 4 FT LOS BARRAZA - ESE HOMBRE
IMAGINE DRAGONS - BELIEVER
ED SHEERAN - SHAPE OF YOU
AVICII - WAKE ME UP
ROMEO SANTOS - PROPUESTA INDECENTE
LADY GAGA - BLOODY MARY
SODA STEREO - DE MUSICA LIGERA
LUIS FONSI - NO ME DOY POR VENCIDO
HARRY STYLES - SIGN OF THE TIMES
COLDPLAY - YELLOW
BOM JOVI- IT'S MY LIFE
FOUDEQUSH CON LA BRISA
TAYLOR SWIFT - SHAKE IT OFF
VICTOR MANUEL - MALA TU
GIAN MARCO GRUPO 5 - EL RITMO DE MI CORAZON
DON OMAR - LOS BANDOLEROS
LOS DAVALOS - MONTONERO AREQUIPEÑO
EVA AYLLON - SACA LAS MANOS
OCOBAMBA -MATADOR
WILLIAN LUNA - NIÑACHAY
LOS KJARKAS - LLORANDO SE FUE
BAD BUNNY- UN VERANO SIN TI
J BALVIN Y MARIA BECERRA - QUE MAS PUES
ROSALIA - DESPECHÁ
DON OMAR - DANZA KUDURO
OZUNA - HEY MOR
BECKY G , CAROL G - MAMIII
RIO ROMA - MI PERSONA FAVORITA
CARLOS BAUTE - COLGANDO EN TUS MANOS
HA ASH - PERDON
JESSE Y JOY - CORRE
REIK - CREO EN TI
""")
st.write("""ROSA PARA CUANDO TU RETO Y LA APUESTA QUE PERDISTE""")

from PIL import Image
image1 = Image.open('descarga.png')

st.image(image1, caption='Sunrise by the mountains')

st.write("""FORMULARIO IMAGENES""")

from PIL import Image
image1 = Image.open('10000.png')

st.image(image1, caption='Sunrise by the mountains')

from PIL import Image
image1 = Image.open('12353.png')

st.image(image1, caption='Sunrise by the mountains')



st.write("""PREPROCESAMIENTO DE DATOS""")



from PIL import Image
image1 = Image.open('23333.png')








st.write("""LECTURA DE TABLA PRINCIPAL""")
datos=pd.read_csv("ENCUESTA.csv")
datos
st.write("""FILAS X COLUMNAS""")
datos.shape
st.write("""BUSCAR QUE DATOS SON NA""")
nadatos=datos.isnull().sum()
st.write(nadatos)
st.write("""TIPO DE DATOS""")
datos.dtypes
st.write("""IMPUTACION""")
st.write("""DESCRIPCION DE LA TABLA CON LA MEDIA PARA EL METODO DE IMPUTACION""")


describedatos = datos.describe()
st.write(describedatos)




st.write("""REALIZANDO LA IMPUTACION DE DATOS CON LA MEDIA""")

ndatos=datos.fillna({"POR USTEDES":4,"PROPUESTA INDECENTE":4,"SIGN OF THE TIMES ":4,"BLOODY MARY":4,"MY LIFE ":4,"CON LA BRISA ":3,"MALA TU":3,"MONTONERO AREQUIPENO":4,"MATADOR":3,"NINACHAY":5,"UN VERANO SIN TI":3,"QUE MAS PUES ":3,"DESPECHA":3,"DANZA KUDURO":4,"HEY MOR":3,"MAMIII":3,"COLGANDO EN TUS MANOS":4,"CREO EN TI":4})
st.write(ndatos)



st.write("""VERIFICAR IMPUTACION""")

veinputacion=ndatos.isnull().sum()
st.write(veinputacion)


st.write("""CORRELACION DE PEARSON""")


matrizn=n = datos[datos.columns[1:]].to_numpy()
matrizm=m = datos[datos.columns[0]].to_numpy()
st.write(matrizn)
st.write(matrizm)


st.write("""CORRELACION DE pandas""")
pandas=n.T
st.write(pandas)

st.write("""CORRELACION DE pandas""")
dat=df=pd.DataFrame(n.T, columns = m)
st.write(dat)
st.write("""CORRELACION DE pandas""")
datc=m_corr=df.corr()
st.write(datc)
st.write("""MATRIX DE CORRELACION""")
datcc=m_corr_d = np.round(m_corr, 
                       decimals = 2)  
st.write(datcc)



st.write("""MAPA DE CALOR""")


from PIL import Image
image = Image.open('gc1.png')

st.image(image, caption='Sunrise by the mountains')



st.write("""PROPUESTA""")
sn1 = ndatos[ndatos.columns[1:]].to_numpy()
sm1 = ndatos[ndatos.columns[0]].to_numpy()
st.write(sn1)
st.write(sm1)
st.write("""VALIDACION""")
calculo_pearson=[]
def correlax_pearson(x,y):
    mx=x.mean()
    my=y.mean()
    num=np.sum((x-mx)*(y-my))
    den=np.sqrt(np.sum((x-mx)**2)*np.sum((y-my)**2))
    return num / den
for i in range(len(sm1)):
    for j in range(len(sm1)):
        caln=ndatos.loc[[i,j],:]
        ndt=caln[caln.columns[1:]].to_numpy()
        calculo_pearson.append(correlax_pearson(ndt[0],ndt[1]))
rpearson=np.array(calculo_pearson).reshape(len(sm1),len(sm1))
mostrar_=mostrar=pd.DataFrame(rpearson,sm1,sm1)
st.write(mostrar_)
st.write("""MATRIX DE CORRELACION """)


corf_=corf=pd.DataFrame(sn1.T, columns = sm1)
st.write(corf_)
st.write("""MATRIX DE CORRELACION """)
corrmusica_=corrmusica=corf.corr()
st.write(corrmusica_)

st.write("""Matrix correlacion""")
corr_s=corrmusica_spearman=corf.corr(method="spearman")
st.write(corr_s)
st.write("""RESULTADO BUSQUEDA""" )
def correx(corr_mat):
    '''
    Función para convertir una matriz de correlación de pandas en formato tidy.
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['Nombres1','Nombres2','r']
    corr_mat = corr_mat.loc[corr_mat['Nombres1'] != corr_mat['Nombres2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

correx(corrmusica_).head()

st.write("""RESULTADO busqueda""" )
#maximos=resultado.unstack()
#maximos.sort_values(ascending=False)[range(len(n1),((len(n1)+12)))]

st.write("""PRIMER MAXIMO""" )
corrmusica_[corrmusica==1]=0
cf_=primermaximo=corrmusica_.max()
st.write(cf_.max())
st.write(cf_)
st.write("""SEGUNDO MAXIMO""" )
corrmusica[corrmusica==0.9211637210055882]=0
cf2=segundomaximo=corrmusica.max()
st.write(cf2.max())
st.write(cf2)





st.write("""MAPA DE CALOR""")


from PIL import Image
image1 = Image.open('gc2.png')

st.image(image1, caption='Sunrise by the mountains')






#sns.heatmap(resultado,cmap="hot")



#from scipy.spatial.distance import euclidean
#P=int(len(sm1))
#Q=np.zeros((P,P))
#for i in range (P):
 #   for j in range(P):
  #      print("\nEntre",sm1[i],'y' ,sn1[j],'\ntenemos: ')
   #     dist=euclidean(sn1[i],sn1[j])
    #    print('\tDistancia = ',dist)
     #   simil=1/(1+dist)
      #  Q[i,j]=simil
       # print('\tSimlitud = ',simil)
        
st.write("""CASO NETFLIX""" )
datos.loc[[0,1],["POR USTEDES"]]
ndatos.loc[[0,1],["POR USTEDES"]]
ndatos.loc[[0,1],["POR USTEDES","ME VAS A EXTRANAR"]]
st.write("""RESULTADOS""" )
st.write(""" 1. _noeliaparedesgu@gmail.com	_ y _gparedesg@unsa.edu.pe_  obtienen el **PRIMER** indice mas alto de similitud 
 
 2. _noeliaparedesgu@gmail.com_ y _elhuamani@unsa.edu.pe_ obtienen el **SEGUNDO** indice mas alto de similitud """ )
st.write("""Conclusiones""" )
st.write("""Se importo los datos obtenidos del formulario mediante el formato csv el cual nos permitio la lectura de datos del formulario delimitado por comas
Se imputo los datos Nah los cuales no permitian que el algoritmo funcione correctamente
El metodo de imputacion de datos es mediante la media, luego de obtener los datos con la funcion describe() se pueden observar valores excatos los cuales nos ayudan a una imputacion
La necesidad de librerias como el pandas y el numpy nos ayudan a la lectura de datos y a mostrar los graficos¶
Se delimito la importancia de imputacion de datos y la distancia euclidiana""" )


st.write("""PROYECTO FINAL""")
##df=pd.read_csv("ENCUESTA.csv")
st.line_chart(ndatos)


st.write("""REFERENCIAS""" )
st.write("""Profesor de Matematicas: John Gabriel Muñoz Cruz https://www.linkedin.com/in/jgmc
""" )
st.write("""

https://aprendeconalf.es/docencia/python/manual/numpy/#:~:text=NumPy%20es%20una%20librer%C3%ADa%20de,un%20gran%20volumen%20de%20datos.

https://doc.arcgis.com/es/insights/latest/create/heat-chart.htm

https://datascience.eu/es/matematica-y-estadistica/que-es-una-matriz-de-correlacion/

https://www.lifeder.com/distancia-euclidiana/

https://tudashboard.com/que-es-un-dashboard/

https://www.cursosgis.com/que-es-streamlit/

https://rockcontent.com/es/blog/framework/""" )
